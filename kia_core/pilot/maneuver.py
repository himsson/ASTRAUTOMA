"""Работа с манёвренными узлами: расчёт, ориентация, исполнение ожога."""
from __future__ import annotations

import math
import time

from ..config import CONFIG, G0
from ..logging_setup import get_logger

log = get_logger("pilot.maneuver")


class ManeuverExecutor:
    """Создаёт и исполняет манёвренные узлы с обратной связью по остатку dV."""

    def __init__(self, connection, telemetry, control):
        self.connection = connection
        self.telemetry = telemetry
        self.control = control
        self.sc = connection.space_center

    # ------------------------------------------------------------------
    @property
    def vessel(self):
        return self.telemetry.vessel

    def clear_nodes(self) -> None:
        try:
            for node in list(self.vessel.control.nodes):
                node.remove()
        except Exception as exc:
            log.debug("Не удалось удалить узлы: %s", exc)

    def add_node(self, ut: float, prograde: float = 0.0, normal: float = 0.0,
                 radial: float = 0.0):
        node = self.vessel.control.add_node(ut, prograde=prograde,
                                            normal=normal, radial=radial)
        log.info("Создан узел на UT=%.0f: dV=%.1f м/с (pro=%.1f, nor=%.1f, rad=%.1f)",
                 ut, node.delta_v, prograde, normal, radial)
        return node

    # ------------------------------------------------------------------
    # Плотности топлива KSP, кг на единицу. Нужны, чтобы перевести запас
    # ступени в массу и узнать, докуда её хватит.
    FUEL_DENSITY = {"LiquidFuel": 5.0, "Oxidizer": 5.0,
                    "SolidFuel": 7.5, "MonoPropellant": 4.0}

    def _burn_segments(self) -> list[tuple[float, float, float]]:
        """Цепочка «тяга, Isp, запас топлива» в порядке срабатывания ступеней.

        Возвращает [(Н, с, кг), ...] от нынешней ступени к верхним.
        """
        v = self.vessel
        by_stage: dict[int, list] = {}
        try:
            engines = list(v.parts.engines)
            current = v.control.current_stage
        except Exception:
            return []
        for e in engines:
            try:
                stage = e.part.stage
                if stage > current:
                    continue
                thrust = e.available_thrust or e.max_thrust
                isp = e.specific_impulse or e.vacuum_specific_impulse
                if not thrust or not isp:
                    continue
                by_stage.setdefault(stage, []).append(
                    (thrust, isp, e.part.decouple_stage))
            except Exception:
                continue
        out: list[tuple[float, float, float]] = []
        for stage in sorted(by_stage, reverse=True):
            group = by_stage[stage]
            thrust = sum(t for t, _, _ in group)
            # Общий Isp связки: тяга, делённая на суммарный расход.
            flow = sum(t / (i * G0) for t, i, _ in group)
            isp = thrust / (flow * G0) if flow > 0 else 0.0
            mass = 0.0
            for dstage in {d for _, _, d in group}:
                try:
                    res = v.resources_in_decouple_stage(stage=dstage,
                                                        cumulative=False)
                    mass += sum(res.amount(n) * d
                                for n, d in self.FUEL_DENSITY.items())
                except Exception:
                    continue
            if thrust > 0 and isp > 0:
                out.append((thrust, isp, mass))
        return out

    def burn_time_for(self, delta_v: float) -> float:
        """Время ожога с учётом ступеней, которые выгорят по дороге.

        ОДНОЙ ТЕКУЩЕЙ ТЯГИ МАЛО. Раньше время считалось по тяге на момент
        расчёта, и для длинного ожога это врало втрое: скругление у Муны
        начинал 60-килоньютонный «Терьер», а дожигал 20-килоньютонный
        «Искра». Оценка давала 20 с, ожог шёл 60 — аппарат проходил
        апоапсис ПОСРЕДИ ожога, и последние метры в секунду жглись уже на
        спуске. Орбита выходила 94×64 вместо 80×80.
        """
        need = abs(delta_v)
        try:
            m0 = self.vessel.mass * 1000.0            # кг
        except Exception:
            return 0.0
        segments = self._burn_segments()
        if not segments:
            return float("inf")
        total = 0.0
        for thrust, isp, fuel in segments:
            ve = isp * G0
            flow = thrust / ve                        # кг/с
            if flow <= 0:
                continue
            # Сколько даст эта ступень, если сжечь её досуха
            if fuel > 0 and m0 - fuel > 0:
                dv_segment = ve * math.log(m0 / (m0 - fuel))
            else:
                dv_segment = 0.0
            if dv_segment >= need:
                burned = m0 - m0 / math.exp(need / ve)
                return total + burned / flow
            total += fuel / flow
            m0 -= fuel
            need -= dv_segment
        # Топлива на весь импульс не хватает: возвращаем время до сухих
        # баков — ожог всё равно оборвётся раньше.
        return total

    # ------------------------------------------------------------------
    def orient_to_node(self, node, timeout: float = 120.0) -> bool:
        ap = self.control.ap
        try:
            ap.reference_frame = node.reference_frame
            ap.target_direction = (0.0, 1.0, 0.0)
            self.control.engage_autopilot()
        except Exception as exc:
            log.error("Ориентация на узел не удалась: %s", exc)
            return False
        return self.control.wait_for_orientation(tolerance=3.0, timeout=timeout)

    def warp_to_node(self, node, burn_time: float, lead: float = 0.5) -> None:
        """Варп к моменту начала ожога (с запасом на разворот)."""
        start_ut = node.ut - burn_time * lead
        margin = 15.0
        if start_ut - margin > self.sc.ut:
            self.control.warp_to(start_ut - margin)

    # ------------------------------------------------------------------
    # ДОПУСК ОЖОГА — АБСОЛЮТНЫЙ, А НЕ ДОЛЯ ИМПУЛЬСА.
    #
    # Доля 0.15 означала «пятнадцать процентов узла не отрабатываем
    # никогда». Замер живого полёта (запуск с рекордом 726): узел
    # скругления 99.9 м/с, отработано 86, остаток 14.3 — ожог оборвался
    # сам, с пометкой «добрал». Апоапсис 78.5 км, перицентр 61.8 км:
    # орбита не замкнулась ровно на эти недоданные метры в секунду.
    #
    # Полметра в секунду двигатель добирает на дросселе: ниже 20 м/с
    # остатка тяга уже режется пропорционально (см. тонкую доводку). А
    # от бесконечной погони за нулём страхуют три уже имеющихся выхода —
    # окно `deadline`, счётчик `stall_ticks` и проверка на рост остатка.
    BURN_TOLERANCE = 0.5              # м/с

    # Ошибка наведения, выше которой газ давать нельзя: тяга мимо цели
    # не только тратит топливо впустую, но и раскручивает аппарат.
    BURN_POINT_LIMIT = 20.0           # градусов

    def _hold_pointing(self, restore: float) -> bool:
        """Режет газ, если нос ушёл с цели, и ждёт возврата.

        ПРОВЕРКИ ПЕРЕД ОЖОГОМ МАЛО. Сторож поймал раскрутку уже В ХОДЕ
        ожога: на входе наведение было чистым, а через минуту газ 0.75
        при ошибке 92° и вращении 57 °/с. Длинный ожог смещает центр масс
        и выедает баки, автопилот отстаёт — и тяга сама раскручивает
        аппарат дальше. Возвращает True, если газ пришлось снимать.
        """
        if self._pointing_error() <= self.BURN_POINT_LIMIT:
            return False
        self.control.set_throttle(0.0)
        self.control.set_rcs(True)
        log.warning("Ожог приостановлен: ошибка наведения %.0f° — "
                    "гашу вращение", self._pointing_error())
        # Ждём недолго: цель ожога уплывает, пока мы разворачиваемся.
        self.control.wait_for_orientation(tolerance=self.BURN_POINT_LIMIT,
                                          timeout=20.0)
        self.control.set_throttle(restore)
        return True

    def _pointing_error(self) -> float:
        """Текущая ошибка наведения в градусах (0, если спросить не у кого)."""
        try:
            error = float(self.control.ap.error)
        except Exception:
            return 0.0
        return error if math.isfinite(error) else 0.0

    def execute(self, node, lead: float = 0.5,
                tolerance: float = BURN_TOLERANCE,
                max_duration: float | None = None, autostage: bool = True) -> bool:
        """Исполняет узел. `tolerance` — остаток в м/с, а не доля узла."""
        total_dv = node.delta_v
        if total_dv <= 0.01:
            node.remove()
            return True

        burn = self.burn_time_for(total_dv)
        if not math.isfinite(burn):
            log.error("Нет тяги — узел исполнить невозможно")
            return False
        log.info("Ожог: dV=%.1f м/с, расчётное время %.1f с", total_dv, burn)

        if not self.orient_to_node(node):
            log.warning("Наведение на узел неточное, продолжаем")
        self.warp_to_node(node, burn, lead)
        # Доводим ориентацию после варпа
        self.orient_to_node(node, timeout=60.0)

        # УЗЕЛ МОГ ИСЧЕЗНУТЬ, ПОКА МЫ РАЗВОРАЧИВАЛИСЬ.
        #
        # Игра убирает узел, когда его момент пройден или траектория
        # ушла на другой участок. Полёт к Муне из-за этого падал с
        # «Maneuver node has been removed» — прямо здесь, на чтении
        # `node.ut`. Ронять миссию из-за пропавшего узла незачем.
        try:
            start_ut = node.ut - burn * lead
        except Exception as exc:
            log.warning("Узел исчез до начала ожога: %s", exc)
            return False
        while self.sc.ut < start_ut - 0.1:
            time.sleep(min(0.5, max(0.02, start_ut - self.sc.ut)))

        deadline = time.time() + (max_duration or (burn * 4 + 90))
        initial_remaining = self._remaining(node)
        last_remaining = initial_remaining
        stall_ticks = 0

        # Насколько остаток вправе подрасти, прежде чем считать, что мы
        # проскочили. Пять метров в секунду годятся для доводочного
        # импульса и никуда не годятся для большого: при ожоге на 1744 м/с
        # столько набегает от обычного покачивания при наведении.
        #
        # Замер живого полёта: узел 1744 м/с, отработано 164, остаток 1580,
        # и в журнале ни заглохшего двигателя, ни таймаута — ожог просто
        # оборвался по этому условию через пару секунд после начала.
        overshoot_margin = max(5.0, total_dv * 0.02)
        reason = "дошёл до конца окна"

        # НЕ ЖЕЧЬ, ПОКА НОС НЕ НА УЗЛЕ.
        #
        # Замер сторожа 09:12: газ 0.63, ошибка наведения 89°, вращение
        # 53 °/с тридцать секунд подряд. Двигатель толкал в сторону,
        # раскручивая аппарат ещё сильнее. Результат `orient_to_node`
        # выше читался и выбрасывался — газ подавался безусловно.
        #
        # Противовес: ждём не вечно. Три попытки с гашением вращения
        # через RCS, после чего жжём как есть — недобранный импульс
        # лучше зависшей навсегда миссии.
        for attempt in range(3):
            if self._pointing_error() <= self.BURN_POINT_LIMIT:
                break
            log.warning("Ожог отложен: ошибка наведения %.0f° (попытка %d)",
                        self._pointing_error(), attempt + 1)
            self.control.set_rcs(True)
            self.orient_to_node(node, timeout=45.0)
        self.control.set_throttle(1.0)
        while time.time() < deadline:
            remaining = self._remaining(node)
            if remaining <= max(0.05, tolerance):
                reason = "добрал"
                break
            if remaining > last_remaining + overshoot_margin:
                reason = (f"остаток начал расти ({last_remaining:.0f} -> "
                          f"{remaining:.0f} м/с) — проскочили или сбита "
                          f"ориентация")
                break

            if self._hold_pointing(1.0):
                continue

            # Тонкая доводка на последних метрах в секунду
            if remaining < 20.0:
                self.control.set_throttle(max(0.05, remaining / 20.0))
            else:
                self.control.set_throttle(1.0)

            if autostage and self.control.should_stage():
                self.control.autostage()
                self.orient_to_node(node, timeout=20.0)

            if abs(last_remaining - remaining) < 1e-3:
                stall_ticks += 1
                if stall_ticks > 60:      # ~3 с без прогресса — нет тяги
                    reason = "двигатель не даёт тяги (топливо или стадия)"
                    log.warning("Ожог заглох: остаток %.2f м/с", remaining)
                    break
            else:
                stall_ticks = 0
            last_remaining = remaining
            time.sleep(CONFIG.control.physics_tick)

        self.control.set_throttle(0.0)
        final_remaining = self._remaining(node)
        # Успехом считаем вдвое более широкий допуск, чем цель остановки:
        # иначе ожог, честно доведённый до предела тяги, отмечался бы
        # недобором из-за десятых долей.
        success = final_remaining <= max(1.0, tolerance * 2.0)
        log.info("Ожог завершён. Отработано %.0f из %.0f м/с, остаток %.2f "
                 "(%s) — причина остановки: %s",
                 total_dv - final_remaining, total_dv, final_remaining,
                 "успех" if success else "НЕДОБОР", reason)
        try:
            node.remove()
        except Exception:
            pass
        return success

    def _remaining(self, node) -> float:
        try:
            return node.remaining_delta_v
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Готовые манёвры
    # ------------------------------------------------------------------
    # Сколько раз доводить орбиту и какой прирост перицентра считать
    # осмысленным. Три захода хватает: каждый следующий импульс на
    # порядок меньше предыдущего.
    CIRCULARIZE_ATTEMPTS = 4
    CIRCULARIZE_MIN_GAIN = 100.0      # м прироста перицентра за заход
    # Сколько ждать апоапсиса ради доводочного ожога. Триста секунд было
    # осторожностью после того, как варп на 1337 с увёл игру в ангар. Но
    # замер показал цену этой осторожности: орбита вышла 109×147 км, до
    # зачёта не хватало 28.4 м/с — и доводка была отменена, потому что до
    # апоапсиса оставалось 624 с. Варп с тех пор обёрнут в защиту и
    # миссию уронить не может, так что ждать можно.
    CIRCULARIZE_MAX_WAIT = 1_800.0    # с ожидания апоапсиса ради доводки

    # Насколько поднять перицентр НАД атмосферой, метры. Впритык нельзя:
    # остатки воздуха съедают орбиту за считанные витки.
    PERIAPSIS_CLEARANCE = 5_000.0
    PERIAPSIS_RAISE_TRIES = 6

    def _raise_periapsis_now(self) -> bool:
        """Немедленный разгон, поднимающий перицентр над атмосферой.

        Замер по секундам показал, отчего один заход не замыкает орбиту:
        ожог длится около минуты, аппарат проходит апоапсис ПОСРЕДИ него,
        и последние метры в секунду жгутся уже на спуске.

            t=249  остаток  51.8   Ap=82107  Pe=18732
            t=251  остаток  12.9   Ap=88153  Pe=56072
            t=253  остаток   1.5   Ap=93862  Pe=63874

        Узел отработан весь (остаток 1.5 из 1221), но орбита вышла
        94×64 км — перицентр под атмосферной границей 70 км.

        Возвращает True, если импульс посчитан и выдан.
        """
        v = self.vessel
        body = v.orbit.body
        atmo = body.atmosphere_depth if body.has_atmosphere else 0.0
        target_rp = body.equatorial_radius + atmo + self.PERIAPSIS_CLEARANCE
        mu = body.gravitational_parameter
        done_any = False
        previous = None
        # ИМПУЛЬС СЧИТАЕТСЯ ПО ТЕКУЩЕМУ РАДИУСУ, А ЖЖЁТСЯ ПОЗЖЕ.
        #
        # Пока узел ориентируется и исполняется, аппарат уходит вниз, и
        # точка ожога оказывается не той, для которой считали. Замер:
        # просили 9.6 м/с поднять 63 км до 75 — вышло 67; следом 6.4 м/с
        # на 67 → 75 дали 68.7. Каждый заход недодаёт, но исправно
        # прибавляет, поэтому считаем заново по свежему положению.
        #
        # Противовес: не больше `PERIAPSIS_RAISE_TRIES` заходов, и выход
        # сразу, как только прибавка станет меньше 200 м.
        for _ in range(self.PERIAPSIS_RAISE_TRIES):
            if v.orbit.periapsis >= target_rp:
                break
            ra = v.orbit.apoapsis
            if ra <= target_rp:
                break                       # апоапсис ниже цели — поднимать нечем
            r = v.orbit.radius
            a_new = (ra + target_rp) / 2.0
            under = 2.0 / r - 1.0 / a_new
            if under <= 0.0:
                break
            dv = math.sqrt(mu * under) - math.sqrt(
                mu * (2.0 / r - 1.0 / v.orbit.semi_major_axis))
            if dv <= self.BURN_TOLERANCE:
                break
            log.info("Подъём перицентра на месте: %.1f м/с, чтобы поднять "
                     "%.1f км до %.0f км", dv,
                     v.orbit.periapsis_altitude / 1000.0,
                     (atmo + self.PERIAPSIS_CLEARANCE) / 1000.0)
            self.clear_nodes()
            node = self.add_node(self.sc.ut + 5.0, prograde=dv)
            try:
                self.execute(node, lead=0.0)
            except Exception as exc:
                log.warning("Подъём перицентра прерван: %s", exc)
                break
            done_any = True
            periapsis = v.orbit.periapsis
            if previous is not None and periapsis - previous < 200.0:
                log.info("Подъём перицентра: прибавка иссякла на %.1f км",
                         v.orbit.periapsis_altitude / 1000.0)
                break
            previous = periapsis
        return done_any

    # Длиннее скольких секунд ожог считается «долгим» и исполняется по
    # обратной связи, а не по узлу.
    LONG_BURN_SECONDS = 30.0
    # За сколько секунд до апоапсиса начинать длинный ожог. Он длится
    # около 76 с (замер), так что старт за 45 с кладёт его вокруг
    # апоапсиса, а не после.
    BURN_START_BEFORE_APOAPSIS = 45.0
    # Порог зачёта из `rewards.py`: перицентр выше 70 км, отношение высот
    # больше 0.85. Берём с запасом, чтобы не останавливаться впритык.
    MIN_ORBIT_ALTITUDE = 74_000.0
    CIRCULARITY_GOAL = 0.90
    # Во сколько раз апоапсису позволено вырасти против исходного,
    # прежде чем ожог признаётся убегающим.
    RUNAWAY_FACTOR = 1.6

    def _circularize_feedback(self) -> bool:
        """Скругление по обратной связи: жечь и смотреть на орбиту.

        ПОЧЕМУ НЕ УЗЕЛ. Узел исполняется по РАСЧЁТНОМУ времени ожога, а
        расчёт для этого аппарата не сходится ни в одну сторону. Три
        замера подряд, один и тот же манёвр:

            оценка  20 с, факт ~60 с -> ожог кончился ПОСЛЕ апоапсиса,
                                        орбита  94×64 км
            оценка 214 с, факт ~60 с -> ожог кончился ДО апоапсиса,
                                        орбита 185×72 км

        Между ними — сегментный расчёт по ступеням, который тоже мимо:
        сколько именно даст выгорающая ступень, заранее не знает никто.
        Обратная связь не гадает: держим разгонный вектор в апоапсисе и
        жжём, пока перицентр не поднимется к апоапсису.

        Противовесов три, и любой из них останавливает ожог: перицентр
        достиг цели, перицентр перестал расти (значит топливо кончилось
        или мы промахнулись мимо апоапсиса), истекло окно времени.
        """
        v = self.vessel
        # СНАЧАЛА РАЗВОРОТ, ПОТОМ ПОДХОД К АПОАПСИСУ.
        #
        # Обратный порядок стоил целого полёта: аппарат подходил к
        # апоапсису и только там начинал разворачиваться, а разворот
        # занимает до сорока пяти секунд. Апоапсис за это время
        # проходился, ожог шёл уже на спуске и поднимал не ту сторону —
        # орбита вышла 69.5×226 км вместо круговой.
        ap = self.control.ap
        try:
            ap.reference_frame = v.orbital_reference_frame
            ap.target_direction = (0.0, 1.0, 0.0)      # разгонный вектор
            self.control.engage_autopilot()
        except Exception as exc:
            log.error("Ориентация на разгон не удалась: %s", exc)
            return False
        self.control.wait_for_orientation(tolerance=5.0, timeout=45.0)

        # Теперь ждём апоапсис уже развёрнутыми.
        remaining = v.orbit.time_to_apoapsis - self.BURN_START_BEFORE_APOAPSIS
        if remaining > 5.0:
            self.control.warp_to(self.sc.ut + remaining)
        while v.orbit.time_to_apoapsis > self.BURN_START_BEFORE_APOAPSIS:
            time.sleep(CONFIG.control.physics_tick)

        # ЦЕЛЬ БЕРЁТСЯ ОДИН РАЗ И БОЛЬШЕ НЕ МЕНЯЕТСЯ.
        #
        # Считать её от ТЕКУЩЕГО апоапсиса нельзя: пока идёт ожог,
        # апоапсис растёт, цель убегает вместе с ним, и остановиться
        # невозможно. Так аппарат сжёг всё топливо и ушёл с орбиты
        # Кербина совсем — в журнале «84.1 × −1729978.8 км», то есть
        # гипербола, а проверка сочла это успехом: у гиперболы апоапсис
        # отрицателен, и цель схлопнулась до нижнего порога.
        ap_target = v.orbit.apoapsis_altitude
        goal = max(self.MIN_ORBIT_ALTITUDE, ap_target * self.CIRCULARITY_GOAL)
        runaway = ap_target * self.RUNAWAY_FACTOR
        log.info("Скругление по обратной связи: перицентр %.1f км -> цель "
                 "%.1f км при апоапсисе %.1f км",
                 v.orbit.periapsis_altitude / 1000.0, goal / 1000.0,
                 ap_target / 1000.0)
        deadline = time.time() + 300.0
        best = v.orbit.periapsis
        stalled = 0
        self.control.set_throttle(1.0)
        try:
            while time.time() < deadline:
                periapsis = v.orbit.periapsis
                # ЦЕЛЬ ОЖОГА = КРИТЕРИЙ НАГРАДЫ, а не круглое число.
                # Устойчивой орбита считается при перицентре выше 70 км и
                # отношении высот больше 0.85 (`rewards.py`). Прошлый
                # порог был задан в долях РАДИУСА, а полтора процента от
                # 680 км — это десять километров: ожог останавливался на
                # 69.5 км, в полукилометре от зачёта.
                pe_alt = v.orbit.periapsis_altitude
                ap_alt = v.orbit.apoapsis_altitude
                # Апоапсис убежал или орбита разомкнулась — жечь дальше
                # значит уходить от планеты. Немедленно прекращаем.
                if ap_alt <= 0.0 or ap_alt > runaway:
                    log.warning("Скругление: апоапсис ушёл до %.1f км при "
                                "цели %.1f км — ожог прекращён",
                                ap_alt / 1000.0, ap_target / 1000.0)
                    break
                if pe_alt >= goal:
                    log.info("Скругление достигнуто: %.1f×%.1f км "
                             "(отношение %.2f)", pe_alt / 1000.0,
                             ap_alt / 1000.0,
                             min(pe_alt, ap_alt) / max(pe_alt, ap_alt))
                    break
                if self._hold_pointing(1.0):
                    continue
                # Последние метры — на малом газе, иначе проскочим.
                gap = goal - pe_alt
                self.control.set_throttle(1.0 if gap > 5_000.0
                                          else max(0.05, gap / 5_000.0))
                # ПЕРЕКЛЮЧЕНИЕ СТУПЕНЕЙ ОБЯЗАТЕЛЬНО.
                #
                # Без него первый же ожог по обратной связи встал через
                # одиннадцать секунд: перицентр рос с −518 до −441 км,
                # потом ступень выгорела досуха, тяга пропала — и защита
                # от заглохшего двигателя приняла это за конец работы.
                # Узловой ожог стадии переключает, этот обязан тоже.
                if self.control.should_stage():
                    self.control.autostage()
                    try:
                        ap.reference_frame = v.orbital_reference_frame
                        ap.target_direction = (0.0, 1.0, 0.0)
                    except Exception:
                        pass
                    self.control.set_throttle(1.0)
                    stalled = 0
                    continue
                if periapsis > best + 50.0:
                    best = periapsis
                    stalled = 0
                else:
                    stalled += 1
                    if stalled > 60:          # ~3 с без роста
                        log.info("Скругление: перицентр не растёт на %.1f км "
                                 "— ожог прекращён",
                                 v.orbit.periapsis_altitude / 1000.0)
                        break
                time.sleep(CONFIG.control.physics_tick)
        finally:
            self.control.set_throttle(0.0)
        # Итог считаем по тому же критерию, что и награда: высоты, а не
        # радиусы, и отношение — а не одна лишь высота перицентра.
        pe_alt = v.orbit.periapsis_altitude
        ap_alt = v.orbit.apoapsis_altitude
        if pe_alt <= 0 or ap_alt <= 0:
            return False
        ratio = min(pe_alt, ap_alt) / max(pe_alt, ap_alt)
        return pe_alt >= self.MIN_ORBIT_ALTITUDE and ratio >= 0.85

    def circularize_at_apoapsis(self, lead: float = 0.5) -> bool:
        """Скругляет орбиту, при нужде в несколько заходов.

        ОДНОГО ЗАХОДА НЕ ХВАТАЕТ, И ЭТО НЕ ОШИБКА НАВЕДЕНИЯ.
        Узел считается как мгновенный импульс, а исполняется реальным
        ожогом. Замер живого полёта к Муне: узел 1179 м/с, расчётное
        время 20 с, фактическое 97 с — посреди ожога отделяется ступень,
        и остаток дожигает двигатель втрое слабее. За полторы минуты
        вокруг апоапсиса вектор тяги успевает уйти, импульс
        «размазывается»:

            Отработано 1178 из 1179 м/с (успех)
            Скругление неполное: Ap=93915 Pe=63782

        Формально ожог выполнен весь, а перицентр остался под
        атмосферой — и программа сдавалась при полных баках. Второй
        заход считает узел уже от новой орбиты, и импульс там на порядок
        меньше, то есть короткий и почти мгновенный.

        Противовес: заходов не больше `CIRCULARIZE_ATTEMPTS`, и цикл
        обрывается, как только перицентр перестаёт заметно расти —
        топливо не сжигается впустую.
        """
        best_periapsis = None
        for attempt in range(1, self.CIRCULARIZE_ATTEMPTS + 1):
            v = self.vessel
            mu = v.orbit.body.gravitational_parameter
            r = v.orbit.apoapsis
            a1 = v.orbit.semi_major_axis
            # НА РАЗОМКНУТОЙ ОРБИТЕ АПОАПСИСА НЕТ.
            #
            # У гиперболы `apoapsis` отрицателен, и подкоренное выражение
            # уходит в минус: полёт падал с ValueError (−2042) прямо в
            # расчёте скругления. Считать здесь нечего — уходим.
            if r <= 0.0 or v.orbit.eccentricity >= 1.0:
                log.warning("Орбита разомкнута (эксцентриситет %.2f) — "
                            "скруглять нечего", v.orbit.eccentricity)
                return False
            under = 2.0 / r - 1.0 / a1
            if under <= 0.0:
                log.warning("Скругление: расчёт невозможен на этой орбите")
                return False
            v1 = math.sqrt(mu * under)
            v2 = math.sqrt(mu / r)
            dv = v2 - v1
            if abs(dv) <= self.BURN_TOLERANCE:
                return True
            # Короткий импульс узел отрабатывает точно — так лёгкая
            # двухступенчатая ракета берёт орбиту Кербина, и ломать это
            # незачем. Долгий — только по обратной связи.
            if attempt == 1 and self.burn_time_for(dv) > self.LONG_BURN_SECONDS:
                if self._circularize_feedback():
                    return True
                continue
            if attempt > 1:
                # ВЫСОТЫ, А НЕ РАДИУСЫ. `orbit.apoapsis` — расстояние от
                # ЦЕНТРА тела; на Кербине это на 600 км больше высоты.
                # Ровно на этом я и обманулся: орбита 94×64 км выглядела в
                # журнале как 694×663 и была принята за устойчивую.
                log.info("Скругление, заход %d: остаточный импульс %.1f м/с "
                         "(Ap=%.0f км, Pe=%.0f км)", attempt, dv,
                         v.orbit.apoapsis_altitude / 1000.0,
                         v.orbit.periapsis_altitude / 1000.0)
            # ДОВОДКА НЕ СТОИТ ПОЛУЧАСОВОГО ВАРПА.
            #
            # Замер: первый ожог перелетел и оставил аппарат на почти
            # круговой орбите 694×663 км. Остаточный импульс — всего
            # 25.5 м/с, но до апоапсиса такой орбиты 1337 с, и варп на
            # них увёл игру в ангар посреди полёта:
            #
            #     Варп не удался: Object reference not set...
            #     Сбой миссии: Procedure not available in scene 'EditorVAB'
            #
            # Орбита при этом была УЖЕ устойчивой. Гнаться за точной
            # высотой, рискуя всем полётом, незачем.
            if attempt > 1 and v.orbit.time_to_apoapsis > self.CIRCULARIZE_MAX_WAIT:
                # Ждать апоапсиса больше двадцати минут нельзя: варп на
                # такую даль уже уводил игру в ангар посреди полёта. Но и
                # бросать орбиту с перицентром в атмосфере незачем —
                # перицентр поднимает ЛЮБОЙ разгон вдали от него, просто
                # чуть дороже, чем в апоапсисе. Жжём сразу.
                if self._raise_periapsis_now():
                    continue
                log.info("Скругление: до апоапсиса %.0f с — доводка "
                         "отменена, орбита %.0f×%.0f км",
                         v.orbit.time_to_apoapsis,
                         v.orbit.periapsis_altitude / 1000.0,
                         v.orbit.apoapsis_altitude / 1000.0)
                break
            ut = self.sc.ut + v.orbit.time_to_apoapsis
            self.clear_nodes()
            node = self.add_node(ut, prograde=dv)
            try:
                self.execute(node, lead=lead)
            except Exception as exc:
                # Сцена могла смениться, судно — исчезнуть. Это не повод
                # ронять всю миссию из доводочного ожога.
                log.warning("Скругление, заход %d прерван: %s", attempt, exc)
                break

            periapsis = self.vessel.orbit.periapsis
            if best_periapsis is not None and \
                    periapsis - best_periapsis < self.CIRCULARIZE_MIN_GAIN:
                log.info("Скругление: перицентр перестал расти "
                         "(%.0f -> %.0f м) — заходы прекращены",
                         best_periapsis, periapsis)
                best_periapsis = max(best_periapsis, periapsis)
                break
            best_periapsis = periapsis
        # Успех решает не число заходов, а форма орбиты: её проверяет
        # вызывающая сторона по достижению stable_orbit.
        v = self.vessel
        return abs(v.orbit.apoapsis - v.orbit.periapsis) < max(
            2_000.0, v.orbit.apoapsis * 0.02)

    def circularize_at_periapsis(self, lead: float = 0.5) -> bool:
        v = self.vessel
        mu = v.orbit.body.gravitational_parameter
        r = v.orbit.periapsis
        a1 = v.orbit.semi_major_axis
        v1 = math.sqrt(mu * (2.0 / r - 1.0 / a1))
        v2 = math.sqrt(mu / r)
        dv = v2 - v1
        ut = self.sc.ut + v.orbit.time_to_periapsis
        self.clear_nodes()
        node = self.add_node(ut, prograde=dv)
        return self.execute(node, lead=lead)

    def change_apoapsis(self, target_apoapsis_alt: float, lead: float = 0.5) -> bool:
        """Ожог в периапсисе для подъёма/опускания апоапсиса."""
        v = self.vessel
        body = v.orbit.body
        mu = body.gravitational_parameter
        rp = v.orbit.periapsis
        ra_target = body.equatorial_radius + target_apoapsis_alt
        a_target = (rp + ra_target) / 2.0
        v_now = math.sqrt(mu * (2.0 / rp - 1.0 / v.orbit.semi_major_axis))
        v_target = math.sqrt(mu * (2.0 / rp - 1.0 / a_target))
        ut = self.sc.ut + v.orbit.time_to_periapsis
        self.clear_nodes()
        node = self.add_node(ut, prograde=v_target - v_now)
        return self.execute(node, lead=lead)

    def retrograde_burn(self, delta_v: float, ut: float | None = None,
                        lead: float = 0.5) -> bool:
        ut = ut if ut is not None else self.sc.ut + 10.0
        self.clear_nodes()
        node = self.add_node(ut, prograde=-abs(delta_v))
        return self.execute(node, lead=lead)
