"""Перелёт к Муне: расчёт окна, узел инжекции, коррекция, захват.

Алгоритм:
1. По фазовому углу Гомана ищем UT, когда цель займёт нужное положение.
2. Строим узел с расчётным dV инжекции.
3. Уточняем узел численно (координатный спуск по dV и UT) до появления
   перехвата SOI Муны с нужным периапсисом — используем патчи kRPC.
4. Исполняем ожог, ждём смены SOI, тормозим в периапсисе Муны.
"""
from __future__ import annotations

import math
import time

from ..engineer import rocket_math as rm
from ..engineer.parts_db import get_body
from ..learning.rewards import Failure, Milestone, RewardSystem
from ..logging_setup import get_logger

log = get_logger("pilot.transfer")


class TransferPlanner:
    """Планировщик и исполнитель межпланетного (межлунного) перелёта."""

    def __init__(self, connection, telemetry, control, maneuver,
                 flight_genome, rewards: RewardSystem, target: str = "Mun",
                 target_periapsis: float | None = None, flight_log=None):
        self.connection = connection
        self.telemetry = telemetry
        self.control = control
        self.maneuver = maneuver
        self.g = flight_genome
        self.rewards = rewards
        self.target_name = target
        self.target_periapsis = (target_periapsis if target_periapsis is not None
                                 else flight_genome.mun_periapsis_target)
        self.flight_log = flight_log
        self.sc = connection.space_center

    def _status(self, text: str, detail: str = "") -> None:
        if self.flight_log:
            self.flight_log.status(text, detail)

    def _fact(self, text: str, detail: str = "") -> None:
        if self.flight_log:
            self.flight_log.fact(text, detail)

    # ------------------------------------------------------------------
    @property
    def vessel(self):
        return self.telemetry.vessel

    @property
    def target_body(self):
        return self.sc.bodies[self.target_name]

    # ------------------------------------------------------------------
    def phase_angle_now(self) -> float:
        """Текущий фазовый угол цели относительно судна, радианы [0, 2π)."""
        parent = self.vessel.orbit.body
        ref = parent.non_rotating_reference_frame
        vp = self.vessel.position(ref)
        tp = self.target_body.position(ref)
        a = math.atan2(vp[2], vp[0])
        b = math.atan2(tp[2], tp[0])
        return (b - a) % (2 * math.pi)

    def required_phase_angle(self) -> float:
        parent = self.vessel.orbit.body
        mu = parent.gravitational_parameter
        r1 = self.vessel.orbit.semi_major_axis
        r2 = self.target_body.orbit.semi_major_axis
        a_t = (r1 + r2) / 2.0
        tof = math.pi * math.sqrt(a_t ** 3 / mu)
        omega_t = 2 * math.pi / self.target_body.orbit.period
        angle = math.pi - omega_t * tof
        return (angle + math.radians(self.g.transfer_phase_bias)) % (2 * math.pi)

    # ------------------------------------------------------------------
    # МЕЖПЛАНЕТНОЕ ОКНО — СОВСЕМ ДРУГАЯ ЗАДАЧА, ЧЕМ ЛУННОЕ.
    #
    # Прежний расчёт считал угол между АППАРАТОМ и целью вокруг одного
    # тела. К Муне это верно: и аппарат, и Муна ходят вокруг Кербина.
    # К Дюне — бессмыслица: она ходит вокруг Солнца, и сравнивать её
    # положение с положением аппарата на орбите Кербина не с чем.
    #
    # Замер живого полёта: расчёт выдал «окно через 34 минуты» (у Дюны
    # оно наступает раз в пару лет), уточнение честно доложило «перехват
    # не найден», а ожог всё равно ушёл на 915 м/с в пустоту.
    #
    # Правильная задача — гомановский перелёт между ПЛАНЕТАМИ:
    #   время перелёта      t = π √(a³/μ☉),  a = (r₁+r₂)/2
    #   нужный угол         θ = π − ω₂·t     (цель должна быть впереди)
    #   ждать               (θ − θ_сейчас) / (ω₂ − ω₁)
    # ------------------------------------------------------------------
    def _is_interplanetary(self) -> bool:
        """Цель вращается вокруг другого тела, чем ДОМ.

        СРАВНИВАТЬ НАДО С ДОМОМ, А НЕ С ТЕКУЩИМ ТЕЛОМ.
        Прежде признак брался как «родитель цели ≠ тело, вокруг которого
        мы летим сейчас». На опорной орбите это верно, но сразу после
        инжекции аппарат летит уже вокруг СОЛНЦА — то есть вокруг того
        же тела, что и Дюна, — и перелёт объявлялся лунным. Пилот шёл по
        короткой ветке, ждал предсказанной встречи и прерывал миссию,
        хотя коррекция довела промах до 42 км.
        """
        try:
            home = getattr(self, "_home_name", None)
            if home is None:
                home = self.vessel.orbit.body.name
                self._home_name = home
            return self.target_body.orbit.body.name != home
        except Exception:
            return False

    def _planet_angle(self, body) -> float:
        """Угол планеты на орбите вокруг светила, радианы."""
        star = body.orbit.body
        frame = star.non_rotating_reference_frame
        x, _, z = body.position(frame)
        return math.atan2(z, x)

    def _transfer_time(self) -> float:
        """Время гомановского перелёта до цели, секунды."""
        home = self.vessel.orbit.body
        if home.orbit is None:
            return 0.0
        star = home.orbit.body
        a = (home.orbit.semi_major_axis
             + self.target_body.orbit.semi_major_axis) / 2.0
        return math.pi * math.sqrt(a ** 3 / star.gravitational_parameter)

    def interplanetary_window(self) -> tuple[float, float]:
        """(секунды ожидания, импульс схода) для перелёта к другой планете."""
        home = self.vessel.orbit.body           # Кербин
        target = self.target_body               # Дюна
        star = home.orbit.body                  # Кербол
        mu_star = star.gravitational_parameter
        r1 = home.orbit.semi_major_axis
        r2 = target.orbit.semi_major_axis

        a_t = (r1 + r2) / 2.0
        t_transfer = math.pi * math.sqrt(a_t ** 3 / mu_star)

        omega1 = 2.0 * math.pi / home.orbit.period
        omega2 = 2.0 * math.pi / target.orbit.period
        # Цель обязана стоять ВПЕРЕДИ на столько, сколько сама пройдёт за
        # время перелёта, — иначе мы придём в точку, где её уже нет.
        required = math.pi - omega2 * t_transfer
        current = self._planet_angle(target) - self._planet_angle(home)

        rate = omega2 - omega1                  # для внешней планеты < 0
        delta = (required - current) % (2.0 * math.pi)
        if rate < 0:
            delta -= 2.0 * math.pi              # ждём, пока догоним «назад»
        wait = delta / rate if rate else 0.0
        if wait < 0:
            wait += abs(2.0 * math.pi / rate)   # следующее окно

        # Импульс схода: сперва скорость на бесконечности из гомановской
        # орбиты вокруг светила, затем разгон с опорной орбиты Кербина.
        v_home = math.sqrt(mu_star / r1)
        v_peri = math.sqrt(mu_star * (2.0 / r1 - 1.0 / a_t))
        v_inf = abs(v_peri - v_home)
        mu_home = home.gravitational_parameter
        r_park = self.vessel.orbit.semi_major_axis
        dv = (math.sqrt(v_inf * v_inf + 2.0 * mu_home / r_park)
              - math.sqrt(mu_home / r_park))

        log.info("Межпланетное окно к %s: перелёт %.1f сут, нужный угол "
                 "%.1f°, сейчас %.1f°, ждать %.1f сут; v∞=%.0f м/с, "
                 "импульс схода %.0f м/с", target.name,
                 t_transfer / 21600.0, math.degrees(required),
                 math.degrees(current), wait / 21600.0, v_inf, dv)
        return wait, dv

    def time_to_window(self) -> float:
        """Секунды до наступления окна перелёта."""
        if self._is_interplanetary():
            return self.interplanetary_window()[0]
        v_period = self.vessel.orbit.period
        t_period = self.target_body.orbit.period
        omega_v = 2 * math.pi / v_period
        omega_t = 2 * math.pi / t_period
        current = self.phase_angle_now()
        target = self.required_phase_angle()
        wait = rm.time_to_phase_angle(current, target, omega_v, omega_t)
        if not math.isfinite(wait):
            return 0.0
        return max(0.0, wait)

    def injection_delta_v(self) -> float:
        if self._is_interplanetary():
            return self.interplanetary_window()[1]
        parent = self.vessel.orbit.body
        mu = parent.gravitational_parameter
        r1 = self.vessel.orbit.semi_major_axis
        r2 = self.target_body.orbit.semi_major_axis
        a_t = (r1 + r2) / 2.0
        v1 = math.sqrt(mu / r1)
        v_peri = math.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
        exact = v_peri - v1
        # Экзамен математика прямо в полёте. Решение принимает формула —
        # сеть отвечает только за то, чтобы её знание было измеримо, и за
        # сигнал о непригодных исходных данных (см. neural/math_brain.py).
        self._audit_math("raise_apoapsis", exact,
                         [mu / 3.5e12, r1 / 1.0e6, r2 / 1.0e6,
                          r2 / max(r1, 1.0) / 15.0, 0, 0])
        return exact

    def _audit_math(self, key: str, exact: float, params: list) -> None:
        """Сверяет ответ сети с формулой и копит статистику. Не влияет на полёт."""
        brain = getattr(self, "_math_brain", None)
        if brain is False:
            return
        if brain is None:
            try:
                from ..neural.math_brain import MathBrain
                brain = MathBrain()
            except Exception:
                self._math_brain = False
                return
            self._math_brain = brain
        try:
            result = brain.verify(key, exact, params)
            if result.get("checked"):
                brain.save()
        except Exception:
            self._math_brain = False

    # ------------------------------------------------------------------
    # Сколько смен сферы влияния просматривать вперёд. К Муне встреча
    # видна через одну; к Дюне их ДВЕ — сперва уход из сферы Кербина в
    # солнечную орбиту, и лишь потом встреча с планетой. Проверка
    # смотрела на один участок, поэтому обход витка честно перебрал
    # тридцать шесть точек и все объявил негодными.
    PATCH_DEPTH = 4

    def _encounter_periapsis(self, node) -> float | None:
        """Периапсис у цели, если он есть на цепочке участков после узла."""
        try:
            patch = node.orbit
            for _ in range(self.PATCH_DEPTH):
                patch = patch.next_orbit
                if patch is None:
                    return None
                if patch.body.name == self.target_name:
                    return patch.periapsis_altitude
            return None
        except Exception:
            return None

    def _solar_patch(self, node):
        """Участок вокруг светила: текущий или после ухода из сферы дома."""
        try:
            home = self.vessel.orbit.body
            # УЖЕ УШЛИ — ЗНАЧИТ УЧАСТОК ПЕРЕД НАМИ, А НЕ ВПЕРЕДИ.
            # После инжекции аппарат сам летит вокруг светила, и поиск
            # «солнечного участка дальше по цепочке» не находил ничего:
            # оценка падала в слепую ветку, и коррекция шла наугад.
            if home.orbit is None or home.orbit.body is None:
                return node.orbit
            star = home.orbit.body.name
            patch = node.orbit
            for _ in range(self.PATCH_DEPTH):
                patch = patch.next_orbit
                if patch is None:
                    return None
                if patch.body.name == star:
                    return patch
        except Exception:
            pass
        return None

    # Сколько точек берём, оценивая сближение. Шестьдесят проб на окно
    # подхода — это шаг в несколько суток на перелёте в триста суток;
    # мельче не нужно, минимум ищется уточнением вокруг лучшей пробы.
    APPROACH_SAMPLES = 60

    def closest_approach(self, orbit, ut_from: float, ut_to: float) -> float:
        """Наименьшее расстояние до цели на участке времени, метры.

        СОВПАДЕНИЕ РАДИУСОВ — НЕ ВСТРЕЧА.
        Прежняя оценка мерила, пересекает ли траектория орбиту цели по
        РАДИУСУ. Замер полёта: коррекции довели этот промах до нуля —
        аппарат приходил на орбиту Дюны идеально, — а встречи не было,
        потому что Дюны в той точке не оказывалось. Видно и по цене:
        последняя правка стоила 97 м/с ради улучшения на 19 км.
        Считать надо расстояние ДО ПЛАНЕТЫ, а не до её орбиты.
        """
        target = self.target_body
        frame = target.orbit.body.non_rotating_reference_frame
        best = float("inf")
        best_ut = ut_from
        span = max(1.0, ut_to - ut_from)
        for i in range(self.APPROACH_SAMPLES + 1):
            ut = ut_from + span * i / self.APPROACH_SAMPLES
            try:
                vp = orbit.position_at(ut, frame)
                tp = target.orbit.position_at(ut, frame)
            except Exception:
                return float("inf")
            d = math.dist(vp, tp)
            if d < best:
                best, best_ut = d, ut
        # Уточняем вокруг лучшей пробы: шаг вдесятеро мельче.
        step = span / self.APPROACH_SAMPLES
        for _ in range(3):
            for ut in (best_ut - step, best_ut + step):
                try:
                    d = math.dist(orbit.position_at(ut, frame),
                                  target.orbit.position_at(ut, frame))
                except Exception:
                    continue
                if d < best:
                    best, best_ut = d, ut
            step /= 4.0
        return best

    def _score_node(self, node) -> float:
        """Чем меньше, тем лучше: 0 — идеальный периапсис у цели."""
        pe = self._encounter_periapsis(node)
        if pe is not None:
            return abs(pe - self.target_periapsis) / 1000.0

        # Встречи ещё нет — меряем настоящее сближение по времени.
        if self._is_interplanetary():
            solar = self._solar_patch(node)
            if solar is not None:
                try:
                    start = node.ut
                    finish = start + self._transfer_time() * 1.4
                    miss = self.closest_approach(solar, start, finish)
                    if math.isfinite(miss):
                        return 1e6 + miss / 1000.0
                except Exception:
                    pass

        # ПЕРЕХВАТА НЕТ — НАДО ХОТЯ БЫ ЗНАТЬ, НАСКОЛЬКО МИМО.
        #
        # Прежде промах мерился апоапсисом ОКОЛОКЕРБИНОВОЙ орбиты против
        # радиуса орбиты цели вокруг Солнца. Величины разного смысла и
        # разного порядка: оценка выходила бессмысленной, и спуск шёл
        # вслепую. Для межпланетного случая смотрим на солнечный участок
        # — до него дотягивается любая разгонная гипербола.
        solar = self._solar_patch(node)
        if solar is not None:
            try:
                target_r = self.target_body.orbit.semi_major_axis
                miss = min(abs(solar.apoapsis - target_r),
                           abs(solar.periapsis - target_r))
                return 1e6 + miss / 1000.0
            except Exception:
                return 1e9
        try:
            ap = node.orbit.apoapsis
            target_r = self.target_body.orbit.semi_major_axis
            return 2e6 + abs(ap - target_r) / 1000.0
        except Exception:
            return 1e9

    def _sweep_ejection(self, node, samples: int = 36):
        """Обходит виток целиком и ставит узел в лучшую точку схода."""
        period = self.vessel.orbit.period
        base_ut = node.ut
        dv = node.prograde
        best_ut, best_score = base_ut, self._score_node(node)
        for i in range(1, samples + 1):
            trial = base_ut + period * i / samples
            try:
                node.ut = trial
            except Exception:
                break
            score = self._score_node(node)
            if score < best_score:
                best_score, best_ut = score, trial
        node.ut, node.prograde = best_ut, dv
        log.info("Точка схода найдена обходом витка: сдвиг %+.0f мин, "
                 "оценка %.0f", (best_ut - base_ut) / 60.0, best_score)
        return node

    def refine_node(self, node, iterations: int = 14):
        """Координатный спуск по (dV, UT) для получения хорошего перехвата.

        ТРЁХ ПРОХОДОВ НЕ ХВАТАЛО, И ЭТО СТОИЛО ВСЕЙ МИССИИ.
        Замер двух живых полётов: спуск останавливался на перицентре
        −35 км и −190 км при заданных 50 км, то есть на траектории
        СКВОЗЬ Муну. Подход выходил почти лобовым, и торможение стоило
        1511 м/с вместо расчётных 298 — впятеро больше бюджета. Топливо
        кончалось, аппарат пролетал мимо.

        Проходы дёшевы: узел не жжёт топливо, игра лишь пересчитывает
        траекторию. Выход по-прежнему ранний — как только перицентр
        сходится к заданному (оценка меньше единицы).
        """
        best_score = self._score_node(node)
        dv = node.prograde
        ut = node.ut
        dv_step = max(20.0, abs(dv) * 0.05)
        side_step = max(5.0, abs(dv) * 0.02)
        ut_step = self.vessel.orbit.period / 12.0

        for it in range(iterations):
            improved = False
            for axis in ("dv", "ut"):
                step = dv_step if axis == "dv" else ut_step
                for direction in (1, -1):
                    trial_dv = dv + (step * direction if axis == "dv" else 0.0)
                    trial_ut = ut + (step * direction if axis == "ut" else 0.0)
                    if trial_ut <= self.sc.ut + 30:
                        continue
                    node.prograde = trial_dv
                    node.ut = trial_ut
                    score = self._score_node(node)
                    if score < best_score - 1e-6:
                        best_score, dv, ut = score, trial_dv, trial_ut
                        improved = True
                        break
                node.prograde, node.ut = dv, ut

            # ПОПЕРЁК ТРАССЫ ОДНИМ ПРОДОЛЬНЫМ ИМПУЛЬСОМ НЕ ПОПРАВИШЬ.
            #
            # Спуск крутил только `prograde` и время. Замер: на 80 % пути
            # расстояние до Дюны вместо убывания выросло с 1.6 до 2.4 млн
            # км — аппарат шёл мимо вбок, а продольная тяга такой промах
            # почти не трогает. Нормаль и радиаль правят именно его.
            if self._is_interplanetary():
                for axis in ("normal", "radial"):
                    base = getattr(node, axis)
                    for direction in (1, -1):
                        setattr(node, axis, base + side_step * direction)
                        score = self._score_node(node)
                        if score < best_score - 1e-6:
                            best_score, base = score, base + side_step * direction
                            improved = True
                            break
                        setattr(node, axis, base)
                    setattr(node, axis, base)
            # ШАГ УМЕНЬШАЕМ ТОЛЬКО КОГДА УПЁРЛИСЬ.
            #
            # Прежде шаг дробился и при удачном проходе тоже, поэтому
            # поиск глох, едва тронувшись: за три прохода шаг падал в
            # десять раз, а до цели оставались сотни километров.
            if not improved:
                dv_step *= 0.5
                side_step *= 0.5
                ut_step *= 0.5
            log.debug("Уточнение %d: dV=%.1f, UT=%.0f, оценка=%.1f", it, dv, ut, best_score)
            if best_score < 1.0:
                break
            if dv_step < 0.5 and ut_step < 1.0:
                break                       # мельче искать бессмысленно

        node.prograde, node.ut = dv, ut
        pe = self._encounter_periapsis(node)
        if pe is not None:
            log.info("Перехват найден: периапсис у %s = %.0f м (dV=%.1f)",
                     self.target_name, pe, dv)
            self.rewards.award(Milestone.MUN_ENCOUNTER,
                               self.telemetry.get("mission_time"),
                               {"periapsis": pe, "dv": dv})
        else:
            log.warning("Перехват не найден, узел оставлен как есть (dV=%.1f)", dv)
        return node

    # ------------------------------------------------------------------
    def plan_transfer(self):
        """Создаёт и уточняет узел перелёта. Возвращает узел или None."""
        self._status(f"Расчёт окна перелёта к {self.target_name}")
        self.control.set_target_body(self.target_name)
        wait = self.time_to_window()

        # СНАЧАЛА ДОЖДАТЬСЯ ОКНА, ПОТОМ СТРОИТЬ УЗЕЛ.
        #
        # К Муне окно наступает через минуты, и порядок не важен. К Дюне
        # ждать 162 суток — а уточнить перехват на таком удалении нельзя
        # в принципе: игра считает встречу по коническим участкам и на
        # 162 суток вперёд её попросту не показывает. Первый полёт из-за
        # этого прервался сразу после расчёта: узел создан правильно,
        # перехвата нет, миссия отменена.
        #
        # Поэтому к дальнему окну сперва подъезжаем варпом, оставляя
        # запас на один виток, и лишь затем считаем заново — уже вблизи,
        # где перехват виден.
        if wait > self.FAR_WINDOW:
            self._status(f"Ожидание окна к {self.target_name}",
                         f"{wait/21600.0:.1f} суток")
            log.info("Варп к окну: %.1f суток вперёд", wait / 21600.0)

            # ПОДХОДИМ СТУПЕНЯМИ, А НЕ ОДНИМ ПРЫЖКОМ.
            #
            # На орбите 800 км игра даёт 100 000×: одна секунда реального
            # времени — больше суток игровых. Запас в один виток (1.7 ч)
            # против такой ставки ничего не значит, и первый же подход
            # проскочил окно на 2°: пришли к углу 42.4° при нужных 44.4°.
            # А следующее окно — через 904 суток, то есть полёт потерян.
            #
            # Убывающий запас решает это без всяких допущений о ставке
            # варпа: сперва встаём за пять суток до окна, потом за сутки,
            # потом за два часа. Каждый шаг медленнее предыдущего, потому
            # что игра сама режет ставку по остатку времени.
            # Ставку варпа режем вместе с запасом: на последних часах
            # 100 000× проскакивает окно раньше, чем игра успевает
            # остановиться — замер дал 43.8° при нужных 44.4°, и
            # следующее окно уехало на 908 суток.
            # Запас первой ступени — тридцать суток: замер показал, что
            # при 100 000× игра успевает проскочить и пятисуточный запас
            # (пришли к 44.0° при нужных 44.4°, окно уехало на 908 суток).
            for margin, rate in ((30 * 21600.0, 100_000.0),
                                 (5 * 21600.0, 10_000.0),
                                 (21600.0, 1_000.0),
                                 (3600.0, 100.0),
                                 (600.0, 50.0)):
                wait = self.time_to_window()
                if wait <= margin:
                    continue
                self.control.warp_to(self.sc.ut + wait - margin, max_rate=rate)
            wait = max(0.0, self.time_to_window())
            log.info("Окно рядом: осталось %.1f ч", wait / 3600.0)

        ut = self.sc.ut + wait
        dv = self.injection_delta_v()
        log.info("Окно перелёта через %.0f с, расчётный dV инжекции %.1f м/с", wait, dv)
        self._fact(f"Окно перелёта найдено: ожидание {wait/60:.1f} мин",
                   f"расчётный импульс {dv:.0f} м/с")

        self.maneuver.clear_nodes()
        try:
            node = self.maneuver.add_node(ut, prograde=dv)
        except Exception as exc:
            log.error("Не удалось создать узел: %s", exc)
            return None

        self.rewards.award(Milestone.MUN_NODE_CREATED,
                           self.telemetry.get("mission_time"),
                           {"dv": round(dv, 1), "ut": round(ut), "wait_s": round(wait)})
        self._fact("Создан манёвренный узел перелёта",
                   f"импульс {dv:.0f} м/с, исполнение через {wait/60:.1f} мин")
        # ГДЕ НА ВИТКЕ ЖЕЧЬ — ОТДЕЛЬНЫЙ ВОПРОС.
        #
        # Угол между планетами говорит, В КАКОЙ ДЕНЬ уходить. Но уходить
        # надо ещё и из нужной ТОЧКИ витка: разгонная гипербола обязана
        # выйти из сферы Кербина вдоль его собственного движения. Замер:
        # к окну подошли идеально (44.6° при нужных 44.4°), а перехвата
        # не нашлось — узел стоял там, где аппарат оказался, а не там,
        # где надо.
        #
        # Виток обходим целиком: пробы топлива не стоят, игра лишь
        # пересчитывает траекторию.
        if self._is_interplanetary():
            node = self._sweep_ejection(node)
        node = self.refine_node(node)

        # НЕ ЖЕЧЬ, ЕСЛИ ПЕРЕХВАТА НЕТ.
        #
        # Замер полёта к Дюне: уточнение честно доложило «перехват не
        # найден», а ожог всё равно ушёл — 915 м/с в пустоту, и аппарат
        # отправился на солнечную орбиту. Предупреждение, после которого
        # всё равно делается по-старому, — это не предупреждение.
        score = self._score_node(node)
        if score >= 1e6 and self._is_interplanetary():
            # ВСТРЕЧУ ИГРА ПОКАЗЫВАЕТ ТОЛЬКО ПРИ БОЛЬШОЙ ТОЧНОСТИ.
            #
            # Замер: обход витка довёл солнечный участок до промаха в
            # 24 тыс. км при радиусе орбиты Дюны 20.7 млн — это 0.12 %,
            # то есть перелёт уже верный. Но патчей встречи игра не
            # рисует, и строгая проверка отменяла безупречный узел.
            #
            # Так же поступают и вручную: уходят по расчёту, а точность
            # добирают коррекцией на середине пути (`correct_course`).
            # Порог держим жёстким — процент радиуса орбиты цели; всё,
            # что грубее, коррекцией уже не спасти.
            miss = (score - 1e6) * 1000.0
            # ПОРОГ МЕРЯЕМ СФЕРОЙ ВЛИЯНИЯ ЦЕЛИ.
            # Теперь `miss` — настоящее расстояние до планеты в момент
            # наибольшего сближения, а не промах мимо её орбиты. Десять
            # сфер влияния Дюны — полмиллиона километров, и это ровно
            # тот порядок, который коррекция в пути уверенно снимает.
            limit = max(self.target_body.sphere_of_influence * 10.0, 1e6)
            if miss <= limit:
                log.info("Встреча не отрисована, но сближение выйдет "
                         "%.0f тыс. км (%.1f сферы влияния) — уходим и "
                         "добираем коррекцией", miss / 1000.0,
                         miss / self.target_body.sphere_of_influence)
                return node
        if score >= 1e6:
            log.warning("Перехвата нет — узел снят, импульс не тратим")
            self._fact("Перелёт отменён: перехват не найден",
                       "жечь топливо в пустоту незачем")
            self.maneuver.clear_nodes()
            return None
        return node

    # ------------------------------------------------------------------
    def execute_transfer(self, node) -> bool:
        self._status("Выполнение перелётного импульса",
                     f"ΔV {node.delta_v:.0f} м/с")
        ok = self.maneuver.execute(node, lead=self.g.node_execute_lead)
        snap = self.telemetry.snapshot()
        self.rewards.evaluate_telemetry(snap)
        if ok:
            self._fact("Перелётный импульс отработан",
                       f"Ap={snap.apoapsis/1000:.0f} км")
        else:
            log.warning("Ожог инжекции неполный")
            if self.flight_log:
                self.flight_log.warning("Перелётный импульс отработан не полностью")
        return ok

    # Потолок коррекции. Это доводка курса, а не второй перелёт: если
    # спуску нужно больше, значит инжекция ушла настолько, что дешевле
    # признать промах, чем сжечь топливо, нужное для торможения у цели.
    COURSE_FIX_CAP = 150.0            # м/с

    # Окно дальше этого срока считается «дальним»: к нему сперва едут
    # варпом, а узел строят уже на месте. Час выбран как заведомо больше
    # любого лунного окна и заведомо меньше любого межпланетного.
    FAR_WINDOW = 3600.0               # секунд

    def correct_course(self) -> bool:
        """Доводит траекторию до перехвата уже после инжекции.

        УЗЕЛ ИМПУЛЬСНЫЙ, А ДВИГАТЕЛЬ — НЕТ. Замер полёта #4: инжекция
        отработана полностью (809 из 810 м/с, остаток 0.47), а до Муны
        осталось 12 213 км. Восемьсот метров в секунду жгутся полторы
        минуты, и всё это время аппарат уходит с точки, для которой узел
        считался. Уточнение ДО ожога этой ошибки не видит по устройству.

        Коррекция стоит копейки: на середине пути метр в секунду двигает
        точку подхода на сотни километров.
        """
        self._status(f"Коррекция курса на {self.target_name}")
        self.maneuver.clear_nodes()
        try:
            node = self.maneuver.add_node(self.sc.ut + 120.0, prograde=0.0)
        except Exception as exc:
            log.error("Узел коррекции не создан: %s", exc)
            return False

        before = self._score_node(node)
        node = self.refine_node(node)
        after = self._score_node(node)
        dv = abs(node.delta_v)

        if after >= before or dv < 0.1:
            log.info("Коррекция не улучшает подход (оценка %.0f -> %.0f) — "
                     "лечу как есть", before, after)
            self.maneuver.clear_nodes()
            return False
        if dv > self.COURSE_FIX_CAP:
            log.warning("Коррекция требует %.0f м/с при потолке %.0f — "
                        "отказываюсь, топливо нужнее на торможение",
                        dv, self.COURSE_FIX_CAP)
            self.maneuver.clear_nodes()
            return False

        log.info("Коррекция курса: %.1f м/с, оценка подхода %.0f -> %.0f",
                 dv, before, after)
        self._fact(f"Коррекция курса на {self.target_name}",
                   f"импульс {dv:.1f} м/с")
        return self.maneuver.execute(node)

    def cruise_to_planet(self, timeout: float = 900.0) -> bool:
        """Перелёт к другой планете: идём ступенями и правим курс.

        ЖДАТЬ ПРЕДСКАЗАННОЙ ВСТРЕЧИ СРАЗУ ПОСЛЕ УХОДА БЕССМЫСЛЕННО.
        Замер: инжекция отработана, коррекция сделана, а `coast_to_soi`
        тут же доложил «смена сферы не предвидится, дистанция 10.6 млн
        км» и прервал миссию. Иначе и быть не могло: до Дюны 302 суток
        пути, и никакой встречи в этот момент игра не показывает.

        Живой порядок другой: идти к цели долями пути, на каждой доле
        смотреть, куда сносит, и подправлять. Ближе к цели точность
        растёт сама, и встреча проявляется.
        """
        star_period = None
        try:
            star_period = self.vessel.orbit.period
        except Exception:
            pass
        deadline = time.time() + timeout
        self._status(f"Перелёт к {self.target_name}")

        # СНАЧАЛА ВЫЙТИ ИЗ СФЕРЫ ДОМА.
        #
        # Сразу после инжекции аппарат ещё внутри сферы Кербина и идёт
        # по ГИПЕРБОЛЕ. У гиперболы нет апоапсиса: `time_to_apoapsis`
        # отдаёт «не число», и расчёт доли пути обрывался на первом же
        # шаге — ступени не включались вовсе.
        home = getattr(self, "_home_name", None)
        for _ in range(3):
            try:
                if self.vessel.orbit.body.name != home:
                    break
                tts = self.vessel.orbit.time_to_soi_change
            except Exception:
                break
            if not math.isfinite(tts) or tts <= 0:
                break
            log.info("Выход из сферы %s через %.1f ч", home, tts / 3600.0)
            self.control.warp_to(self.sc.ut + tts + 60.0)

        for step, fraction in enumerate((0.35, 0.60, 0.80, 0.92, 0.98), 1):
            if time.time() > deadline:
                break
            if self.telemetry.is_in_soi(self.target_name):
                break
            try:
                remaining = self.vessel.orbit.time_to_apoapsis
            except Exception:
                remaining = float("nan")
            if not math.isfinite(remaining) or remaining <= 0:
                break
            self.control.warp_to(self.sc.ut + remaining * fraction)
            pe = self._encounter_periapsis_now()
            log.info("Перелёт: пройдено %.0f%% пути, до цели %.0f тыс. км",
                     fraction * 100.0,
                     self.telemetry.distance_to_body(self.target_name) / 1e6)
            if pe is not None:
                log.info("Встреча появилась: перицентр %.0f км", pe / 1000.0)
                break
            self.correct_course()

        return self.coast_to_soi(timeout=max(60.0, deadline - time.time()))

    def _encounter_periapsis_now(self) -> float | None:
        """Периапсис у цели по текущей траектории (без узла)."""
        try:
            patch = self.vessel.orbit
            for _ in range(self.PATCH_DEPTH):
                patch = patch.next_orbit
                if patch is None:
                    return None
                if patch.body.name == self.target_name:
                    return patch.periapsis_altitude
        except Exception:
            pass
        return None

    def coast_to_soi(self, timeout: float = 900.0) -> bool:
        """Варп до входа в SOI цели с контролем состояния."""
        self._status(f"Перелёт к {self.target_name} (баллистический участок)")
        deadline = time.time() + timeout
        fixed = False
        while time.time() < deadline:
            if self.telemetry.is_in_soi(self.target_name):
                self.telemetry.rebind()
                self.control.rebind()
                self.rewards.award(Milestone.MUN_SOI_ENTERED,
                                   self.telemetry.get("mission_time"))
                log.info("Вошли в сферу влияния %s", self.target_name)
                self._fact(f"Вход в сферу влияния {self.target_name}")
                return True
            try:
                tts = self.vessel.orbit.time_to_soi_change
            except Exception:
                tts = float("nan")
            if math.isnan(tts) or tts <= 0:
                # ПЕРЕД ТЕМ КАК СДАТЬСЯ — ПОПРОБОВАТЬ ДОВЕСТИ КУРС.
                # Одна попытка: если коррекция не помогла, вторая тем
                # более не поможет, а топливо уйдёт.
                if not fixed:
                    fixed = True
                    if self.correct_course():
                        continue
                distance = self.telemetry.distance_to_body(self.target_name)
                self.rewards.shape_target_proximity(distance,
                                                    self.telemetry.get("mission_time"))
                log.warning("Смена SOI не предвидится (мин. дистанция %.0f км)",
                            distance / 1000.0)
                return False
            self.control.warp_to(self.sc.ut + tts + 10.0)
            time.sleep(1.0)
        self.rewards.penalise(Failure.TIMEOUT, self.telemetry.get("mission_time"))
        return False

    # ------------------------------------------------------------------
    def capture(self) -> bool:
        """Торможение в периапсисе Муны до круговой орбиты."""
        if not self.telemetry.is_in_soi(self.target_name):
            log.warning("Захват невозможен: судно вне SOI %s", self.target_name)
            return False

        self._status(f"Торможение у {self.target_name} для выхода на орбиту")
        orbit = self.vessel.orbit
        # Если периапсис слишком низкий — сначала поднимаем его
        target = None                         # the live game has every body; the table does not
        if orbit.periapsis_altitude < 8_000:
            log.info("Периапсис %.0f м слишком низкий — поднимаем", orbit.periapsis_altitude)
            self._raise_periapsis(15_000.0)

        mu = self.vessel.orbit.body.gravitational_parameter
        rp = self.vessel.orbit.periapsis
        v_peri = math.sqrt(mu * (2.0 / rp - 1.0 / self.vessel.orbit.semi_major_axis))
        v_circ = math.sqrt(mu / rp)
        dv = v_circ - v_peri
        ut = self.sc.ut + self.vessel.orbit.time_to_periapsis
        log.info("Захват: dV=%.1f м/с в периапсисе через %.0f с",
                 dv, self.vessel.orbit.time_to_periapsis)

        self.maneuver.clear_nodes()
        node = self.maneuver.add_node(ut, prograde=dv)
        ok = self.maneuver.execute(node, lead=self.g.capture_lead)

        snap = self.telemetry.snapshot()
        self.rewards.evaluate_telemetry(snap)
        self.rewards.shape_orbit_quality(snap.apoapsis, snap.periapsis, snap.mission_time)
        self.control.run_science()
        if Milestone.MUN_ORBIT in self.rewards.achieved:
            log.info("ОРБИТА %s ДОСТИГНУТА: Ap=%.0f Pe=%.0f",
                     self.target_name, snap.apoapsis, snap.periapsis)
            self._fact(f"Выход на орбиту {self.target_name}: "
                       f"{snap.periapsis/1000:.0f}×{snap.apoapsis/1000:.0f} км",
                       f"эксцентриситет {snap.eccentricity:.3f}")
            self.lower_to_target()
            return True
        # СНИЖАТЬ НАДО ПО ФАКТУ ОРБИТЫ, А НЕ ПО ЗАСЧИТАННОЙ ВЕХЕ.
        #
        # Вызов стоял внутри ветки награды, и полёт со счётом 4246 (веха
        # не дана, но орбита замкнута) ушёл на посадку с высокой орбиты —
        # ровно та беда, ради которой снижение и писалось.
        try:
            orb = self.vessel.orbit
            if orb.apoapsis > 0 and orb.periapsis_altitude > 0:
                self.lower_to_target()
        except Exception as exc:
            log.warning("Снижение орбиты не удалось: %s", exc)

        if self.flight_log:
            self.flight_log.warning(f"Захват у {self.target_name} не состоялся",
                                    f"Ap={snap.apoapsis:.0f} м, Pe={snap.periapsis:.0f} м")
        return ok

    def lower_to_target(self) -> bool:
        """Опускает орбиту у цели до заданной высоты двумя импульсами.

        ЗАХВАТ ОСТАНАВЛИВАЛСЯ, ЕДВА ОРБИТА ЗАМКНУЛАСЬ.
        Замер посадки: орбита Муны вышла 1647×1847 км при заданных 50, и
        спуск начинался с такой высоты, что торможение не успевало —
        первая же строка спуска показывала уже отскок обломков. Дело было
        не в опорах и не в расписании снижения, а в том, что садиться
        приходилось с почти половины радиуса сферы влияния.

        Приём обычный, гомановский: в апоцентре тормозим до периапсиса
        нужной высоты, в перицентре скругляем.
        """
        v = self.vessel
        body = v.orbit.body
        mu = body.gravitational_parameter
        r_goal = body.equatorial_radius + self.target_periapsis
        if v.orbit.periapsis <= r_goal * 1.15:
            log.info("Орбита у %s уже низкая (%.0f км) — снижать нечего",
                     self.target_name, v.orbit.periapsis_altitude / 1000.0)
            return True

        self._status(f"Снижение орбиты у {self.target_name}",
                     f"до {self.target_periapsis/1000:.0f} км")
        # Шаг 1: в апоцентре опускаем перицентр до нужной высоты.
        r_ap = v.orbit.apoapsis
        a_new = (r_ap + r_goal) / 2.0
        dv1 = (math.sqrt(mu * (2.0 / r_ap - 1.0 / a_new))
               - math.sqrt(mu * (2.0 / r_ap - 1.0 / v.orbit.semi_major_axis)))
        node = self.maneuver.add_node(self.sc.ut + v.orbit.time_to_apoapsis,
                                      prograde=dv1)
        log.info("Снижение: импульс в апоцентре %.0f м/с (перицентр %.0f -> "
                 "%.0f км)", dv1, v.orbit.periapsis_altitude / 1000.0,
                 self.target_periapsis / 1000.0)
        if not self.maneuver.execute(node):
            log.warning("Импульс снижения не добран")

        # Шаг 2: в перицентре скругляем.
        v = self.vessel
        r_pe = v.orbit.periapsis
        dv2 = (math.sqrt(mu / r_pe)
               - math.sqrt(mu * (2.0 / r_pe - 1.0 / v.orbit.semi_major_axis)))
        node = self.maneuver.add_node(self.sc.ut + v.orbit.time_to_periapsis,
                                      prograde=dv2)
        log.info("Скругление у цели: импульс %.0f м/с", dv2)
        ok = self.maneuver.execute(node)
        v = self.vessel
        log.info("Орбита у %s: %.0f × %.0f км", self.target_name,
                 v.orbit.periapsis_altitude / 1000.0,
                 v.orbit.apoapsis_altitude / 1000.0)
        self._fact(f"Орбита у {self.target_name} снижена",
                   f"{v.orbit.periapsis_altitude/1000:.0f}×"
                   f"{v.orbit.apoapsis_altitude/1000:.0f} км")
        return ok

    def _raise_periapsis(self, target_alt: float) -> bool:
        """Поднимает периапсис у цели поправкой на подлётной траектории.

        В СФЕРУ ВЛИЯНИЯ ВСЕГДА ВХОДЯТ ПО ГИПЕРБОЛЕ, и прежний расчёт
        этого не учитывал: он строил узел «в апоапсисе», а у гиперболы
        апоапсиса нет. `time_to_apoapsis` возвращал бесконечность, и в
        журнале живого полёта это выглядело так:

            Периапсис -189862 м слишком низкий — поднимаем
            Создан узел на UT=inf: dV=125.4 м/с
            Варп на inf с вперёд

        Аппарат шёл на столкновение с Муной, а поправка исполнялась в
        бесконечно далёком будущем, то есть никогда.

        Формулу для эллипса здесь не применить, поэтому поправка
        подбирается численно: узел ставится на ближайшие секунды, а игра
        сама показывает, какая орбита из него получится
        (`node.orbit.periapsis`). Двоичный поиск по боковой составляющей
        находит нужную величину за два десятка проб — без единого ожога.
        """
        v = self.vessel
        body_r = v.orbit.body.equatorial_radius
        target_rp = body_r + target_alt
        # ЗАПАС ВРЕМЕНИ ОБЯЗАТЕЛЕН.
        #
        # Тридцати секунд не хватало: один разворот на подлёте занимает
        # до двух минут («Наведение не сошлось за 120 с»), момент узла
        # проходил, траектория уходила на следующий участок, и игра узел
        # убирала. Полёт падал с «Maneuver node has been removed».
        #
        # Но и тянуть нельзя — поправку надо дать заметно раньше
        # периапсиса, иначе она уже ничего не изменит. Берём меньшее из
        # двух: полторы минуты или треть времени до периапсиса.
        time_to_peri = v.orbit.time_to_periapsis
        if not math.isfinite(time_to_peri) or time_to_peri <= 0:
            time_to_peri = 600.0
        lead_seconds = min(150.0, max(45.0, time_to_peri / 3.0))
        ut = self.sc.ut + lead_seconds
        self.maneuver.clear_nodes()
        node = self.vessel.control.add_node(ut)

        def periapsis_with(radial: float) -> float:
            node.radial = radial
            try:
                return node.orbit.periapsis
            except Exception:
                return float("-inf")

        # Куда крутить: боковая поправка поднимает периапсис в одну
        # сторону и опускает в другую, а какая именно — зависит от того,
        # с какой стороны подлетаем. Пробуем обе.
        probe = 50.0
        if periapsis_with(probe) >= periapsis_with(-probe):
            sign = 1.0
        else:
            sign = -1.0

        # Верхняя граница поиска: наращиваем, пока не перекроем цель.
        high = probe
        for _ in range(6):
            if periapsis_with(sign * high) >= target_rp:
                break
            high *= 2.0
        else:
            log.warning("Периапсис не поднять поправкой до %.0f м/с — "
                        "берём максимум", high)

        low = 0.0
        for _ in range(24):
            mid = (low + high) / 2.0
            if periapsis_with(sign * mid) < target_rp:
                low = mid
            else:
                high = mid
        periapsis_with(sign * high)
        log.info("Поправка подлёта: %.1f м/с вбок — периапсис станет "
                 "%.0f м", sign * high, node.orbit.periapsis - body_r)
        return self.maneuver.execute(node, lead=0.5)

    # ------------------------------------------------------------------
    def run(self) -> bool:
        node = self.plan_transfer()
        if node is None:
            return False
        if not self.execute_transfer(node):
            return False
        if not self.coast_to_soi():
            return False
        return self.capture()
