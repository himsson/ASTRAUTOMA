"""Выведение на орбиту: гравитационный разворот + скругление.

Профиль тангажа:
    pitch = 90° * (1 - ((h - h_start) / (h_end - h_start)) ** n)
где n — ген turn_exponent. Дросселирование ограничивается в зоне
максимального скоростного напора, автостейджинг работает непрерывно.
Каждый шаг пишется в журнал полёта реального времени.
"""
from __future__ import annotations

import math
import time

from ..config import CONFIG
from ..learning.rewards import Failure, Milestone, RewardSystem
from ..logging_setup import get_logger

log = get_logger("pilot.ascent")

# --------------------------------------------------------------------------
# Признаки безнадёжного выведения.
#
# До этого их не было вовсе, и это стоило проекту вечера. Живая ракета с
# показателем кривой тангажа 1.3 шла вертикально вверх, зависала на семи
# километрах и падала обратно — а автопилот полторы минуты рапортовал
# «вывод на орбиту (гравитационный разворот)» и держал нос на 85°. Отказ
# в итоге записывался как out_of_fuel: топливо и правда кончилось, но
# причина была совсем другая, и конструктор из такого полёта не узнавал
# ничего.
#
# Показательно, что ТРЕНАЖЁРЫ опрокидывание ловят давно (sim_env*.py), а
# живая игра — нет. Зеркало истории с пусковыми мачтами, где было ровно
# наоборот: игра знала отказ, которого не знал тренажёр.
# --------------------------------------------------------------------------

# Угол между носом и набегающим потоком, выше которого аппарат считается
# потерявшим устойчивость. 45° — это уже не «сдувает», а «летит боком».
TUMBLE_ANGLE = 45.0
# Сколько секунд подряд угол должен держаться. Одиночный выброс бывает при
# отделении ступени и сам по себе не беда.
TUMBLE_HOLD = 3.0
# Ниже этого скоростного напора поток слабый, и угол атаки ничего не
# решает: в вакууме аппарат может лететь как угодно.
TUMBLE_MIN_Q = 500.0

# --------------------------------------------------------------------------
# ПОВОДОК ПО ПОТОКУ.
#
# Профиль тангажа задаёт УГОЛ ПО ВЫСОТЕ и о том, куда аппарат летит на
# самом деле, не знает ничего. Пока воздух разрежен, это сходит с рук. На
# сверхзвуке в плотных слоях автопилот начинает разворачивать нос против
# набегающего потока, поток подхватывает корпус за бок — и аппарат
# опрокидывает. Оперение тут уже не помогает: замер показал, что
# `fin_level` упирался в потолок 1.0 (четыре стабилизатора), а срывы шли
# по-прежнему, 51-67° от потока на седьмом-десятом километре.
#
# Настоящий гравитационный разворот тем и хорош, что аппарат летит НОСОМ
# ПО ПОТОКУ, а поворачивает его гравитация, а не рули. Поэтому команда
# тангажа теперь держится на коротком поводке от вектора скорости: чем
# плотнее поток, тем короче поводок.
#
# Куда аппарат летит, известно без дополнительных запросов к игре:
# угол вектора скорости = тангаж минус угол атаки.
# --------------------------------------------------------------------------

# Насколько нос вообще разрешено уводить от потока при СИЛЬНОМ напоре.
AIRFLOW_LEASH_MIN = 8.0
# И насколько — когда воздух уже разрежен и опрокинуть нечем.
AIRFLOW_LEASH_MAX = 35.0
# Напор, при котором поводок затягивается до минимума.
AIRFLOW_LEASH_FULL_Q = 12_000.0
# На сколько градусов ниже профиля поводку разрешено опустить команду.
# Дальше — уже не «не спорь с потоком», а «падай вместе с ним».
AIRFLOW_LEASH_BACKSTOP = 20.0
# Отклонение носа от потока, начиная с которого поводок вообще вступает.
# Ниже него аппарат летит носом вперёд, и вмешиваться не во что.
AIRFLOW_LEASH_ACTIVATE = 12.0

# ЗАЩИТА ПО НАПОРУ. Не настройка и не ген: выше этого значения аппарат в
# KSP уже теряет детали от нагрева и перегрузки. Замер живого полёта:
# ракета, свалившаяся в пикирование, дошла до 101 775 Па и едва не
# сгорела, а ограничитель по геному всё это время спокойно отдавал 0.68.
MAX_Q_DANGER = 55_000.0
MAX_Q_DANGER_THROTTLE = 0.15

# Апоапсис, падающий столько секунд подряд при работающем двигателе,
# означает, что аппарат уже не выводится, а тормозится собственным
# профилем. Дальше ждать нечего — только жечь время итерации.
APOAPSIS_DECAY_HOLD = 12.0


class AscentAutopilot:
    """Пилот участка «стартовый стол -> круговая орбита»."""

    def __init__(self, connection, telemetry, control, maneuver,
                 flight_genome, rewards: RewardSystem, target_apoapsis: float,
                 flight_log=None):
        self.connection = connection
        self.telemetry = telemetry
        self.control = control
        self.maneuver = maneuver
        self.g = flight_genome
        self.rewards = rewards
        self.target_apoapsis = target_apoapsis
        self.flight_log = flight_log
        self.sc = connection.space_center
        self.liftoff_ut: float | None = None
        # Состояние детекторов срыва выведения
        self._tumble_since: float | None = None
        self._best_apoapsis: float = 0.0
        self._apoapsis_falling_since: float | None = None
        self._q_warned = False

    # ------------------------------------------------------------------
    def target_pitch(self, altitude: float) -> float:
        g = self.g
        if altitude < g.turn_start_altitude:
            return 90.0
        if altitude >= g.turn_end_altitude:
            return 0.0
        span = max(1.0, g.turn_end_altitude - g.turn_start_altitude)
        frac = (altitude - g.turn_start_altitude) / span
        return 90.0 * (1.0 - frac ** g.turn_exponent)

    def airflow_leash(self, snap, wanted: float) -> float:
        """Не даёт увести нос слишком далеко от вектора скорости.

        Возвращает команду тангажа, урезанную до допустимого отклонения от
        потока. При слабом напоре ограничение почти не мешает — там и
        опрокидывать нечем.
        """
        if not getattr(snap, "aero_angles_known", False):
            return wanted
        q = float(snap.dynamic_pressure or 0.0)
        if q < TUMBLE_MIN_Q:
            return wanted

        # ПОВОДОК ВКЛЮЧАЕТСЯ, ТОЛЬКО ЕСЛИ НОС УЖЕ УШЁЛ ОТ ПОТОКА.
        #
        # Иначе он превращается в постоянный крен вниз. Замер из живой
        # игры показал это прямо:
        #
        #     профиль 90° -> команда 85°   факт 70°
        #     профиль 86° -> команда 66°   факт 48°
        #     профиль 80° -> команда 60°   факт 25°
        #     профиль 77° -> команда 57°   факт  3°
        #
        # Команда всё время на 20° ниже профиля — это упор в собственный
        # ограничитель. Аппарат послушно опускал нос, вектор скорости шёл
        # следом, и ракета сама разворачивала себя к земле. При этом угол
        # атаки был НУЛЕВОЙ: срываться было нечему, оперение держало.
        #
        # То есть поводок боролся с бедой, которой уже нет (её убрали
        # крупные стабилизаторы и ровная стойка на столе), а взамен создал
        # свою. Теперь он молчит, пока нос идёт по потоку, и вмешивается
        # только когда отклонение действительно набралось.
        total = math.hypot(float(snap.angle_of_attack or 0.0),
                           float(getattr(snap, "sideslip_angle", 0.0) or 0.0))
        if total < AIRFLOW_LEASH_ACTIVATE:
            return wanted

        # Куда аппарат летит на самом деле
        flight_path = float(snap.pitch or 0.0) - float(snap.angle_of_attack or 0.0)
        share = min(1.0, q / AIRFLOW_LEASH_FULL_Q)
        leash = AIRFLOW_LEASH_MAX - (AIRFLOW_LEASH_MAX - AIRFLOW_LEASH_MIN) * share
        limited = max(flight_path - leash, min(flight_path + leash, wanted))

        # НИЖНИЙ ПРЕДЕЛ — обязателен, иначе поводок закрепляет срыв.
        #
        # Поводок держит команду у вектора скорости. Но вектор скорости
        # зависит от того, куда аппарат летит, а летит он туда, куда его
        # направили. Если ракета начала заваливаться, вектор уходит вниз,
        # поводок послушно тянет команду за ним, и аппарат заваливается
        # дальше. Положительная обратная связь.
        #
        # Как это выглядело в игре: профиль требовал 89°, поводок отдавал
        # команду 60° на километре и 30° на трёх километрах. Ракета шла
        # почти горизонтально у самой земли и разбилась.
        #
        # Поэтому опускать команду ниже профиля больше чем на BACKSTOP
        # градусов нельзя. Поводок остаётся поводком: он не даёт рулям
        # выворачивать нос ПРОТИВ потока, но и не позволяет молча следовать
        # за падением.
        return max(limited, wanted - AIRFLOW_LEASH_BACKSTOP)

    def throttle_for(self, snap) -> float:
        """Ограничение тяги по скоростному напору + доводка апоапсиса.

        ПОЧЕМУ ЗДЕСЬ НЕ ПРОСТОЙ ПОРОГ.

        Стояло так: напор выше порога — отдать `max_q_throttle` (0.68).
        Обрыв, без возврата и без запаса. Замер живого полёта показал, во
        что это выливается:

            h=1020 м  q= 8 868 Па  тяга полная,  v=179 м/с
            h=3506 м  q=21 916 Па  ТЯГА СРЕЗАНА, v=179 м/с
            h=6006 м  q=29 964 Па  срезана,      v=188 м/с
            h=7276 м  q=38 508 Па  срезана,      апоапсис встал
            h=6529 м  q=65 422 Па  пикирует
            h=3617 м  q=101 775 Па  ракета чуть не сгорела

        Порог перейден один раз — и тяга остаётся 0.68 до конца, потому
        что напор после этого только растёт. TWR падает до 1.2, аппарат
        перестаёт набирать, гравитация заворачивает вектор скорости вниз,
        и ракета идёт за ним носом вперёд (угол атаки при этом нулевой —
        со стороны похоже на потерю управления, но это простая
        баллистика). Дальше пикирование само разгоняет напор впятеро.

        Теперь ограничение ПЛАВНОЕ: тяга снижается по мере приближения к
        порогу, а не обрывается на нём. Аппарат продолжает разгоняться, и
        цепочка «срезали тягу — не набрал — завалился» не запускается.

        И добавлен верхний предел, которого не было вовсе: выше
        `MAX_Q_DANGER` аппарат уже на грани разрушения, и тяга режется
        жёстко независимо от генома. Это не настройка, а защита.
        """
        q = float(snap.dynamic_pressure or 0.0)
        threshold = max(1.0, float(self.g.max_q_threshold))

        if q > MAX_Q_DANGER:
            if not self._q_warned:
                self._q_warned = True
                self._fact("Опасный скоростной напор",
                           f"{q:.0f} Па при пороге разрушения {MAX_Q_DANGER:.0f} — "
                           f"срезаю тягу, аппарат на грани")
                log.error("ОПАСНЫЙ НАПОР %.0f Па на высоте %.0f м — тяга срезана",
                          q, snap.altitude)
            return MAX_Q_DANGER_THROTTLE

        if q > threshold:
            # Плавный спад от полной тяги к `max_q_throttle` на участке
            # от порога до полутора порогов. Дальше — как задано геномом.
            span = threshold * 0.5
            share = min(1.0, (q - threshold) / span)
            floor = float(self.g.max_q_throttle)
            return 1.0 - (1.0 - floor) * share
        if snap.apoapsis > self.target_apoapsis:
            return 0.0
        remaining = self.target_apoapsis - snap.apoapsis
        if remaining < 5_000.0:
            return max(0.15, remaining / 5_000.0)
        return 1.0

    _said_solid = False

    def _solid_burning(self) -> bool:
        """Горит ли сейчас твердотопливный ускоритель.

        Признак — работающий двигатель, у которого нельзя убрать тягу:
        у твердотопливных `throttle_locked` истинно, и команда газа для
        них ничего не значит.

        Ответ живёт секунду: опрос всех двигателей на каждом такте давал
        16 запросов в секунду (rpc_profile.log), а ускоритель за секунду
        не погаснет незаметно.
        """
        now = time.monotonic()
        cached = getattr(self, "_solid_cache", None)
        if cached is not None and now - cached[0] < 1.0:
            return cached[1]
        result = self._solid_burning_now()
        self._solid_cache = (now, result)
        return result

    def _solid_burning_now(self) -> bool:
        try:
            for engine in self.telemetry.vessel.parts.engines:
                if engine.active and engine.has_fuel and engine.throttle_locked:
                    return True
        except Exception:
            pass
        return False

    def _status(self, text: str, detail: str = "") -> None:
        if self.flight_log:
            self.flight_log.status(text, detail)

    def _fact(self, text: str, detail: str = "") -> None:
        if self.flight_log:
            self.flight_log.fact(text, detail)

    # ------------------------------------------------------------------
    def launch(self) -> bool:
        """Зажигание и отрыв. False — если ракета не сдвинулась с места."""
        v = self.telemetry.vessel
        self._status("Предстартовая ориентация",
                     f"курс {self.g.ascent_heading:.0f}°, тангаж 90°")
        self.control.set_sas(False)
        self.control.set_rcs(False)
        self.control.engage_autopilot(v.surface_reference_frame)
        self.control.point(90.0, self.g.ascent_heading, roll=0.0)
        self.control.wait_for_orientation(tolerance=10.0, timeout=15.0)

        # АНТЕННУ НА СТОЛЕ НЕ РАСКРЫВАТЬ.
        #
        # Раскрытая антенна ломается набегающим потоком, и экран
        # результатов полёта говорит это дословно:
        #
        #     [00:00:04] Коммунотрон 16: Антенна разрушается
        #                из-за сопротивления воздуха
        #
        # Четвёртая секунда подъёма. Дальше беспилотный аппарат летит
        # без связи, а без связи игра не даёт им управлять: торможение у
        # Муны срывалось с диагнозом «сбита ориентация», и оператор видел
        # в игре «Нет связи с зондом».
        #
        # Раскрытие перенесено за атмосферу, к солнечным панелям.
        self._status("Включение двигателей 1 ступени")
        # ВАЖНО: до активации ступени available_thrust = 0, поэтому TWR
        # измеряется только ПОСЛЕ зажигания.
        self.control.ignite(1.0)
        self.liftoff_ut = self.sc.ut
        time.sleep(0.6)

        twr = self.telemetry.current_twr()
        if twr <= 0.01:
            # первая стадия могла лишь отпустить крепления — пробуем следующую
            log.info("Тяги нет после первой стадии — активируем следующую")
            self.control.next_stage()
            time.sleep(0.8)
            twr = self.telemetry.current_twr()

        if twr < 1.05:
            log.error("Стартовый TWR=%.2f — ракета не оторвётся", twr)
            self.rewards.penalise(Failure.NO_LIFTOFF, self.telemetry.get("mission_time"),
                                  {"twr": round(twr, 2),
                                   "thrust_kn": round(self.telemetry.get("available_thrust")/1000.0, 1)})
            self.control.set_throttle(0.0)
            return False
        log.info("Тяга есть: TWR=%.2f", twr)

        deadline = time.time() + 25.0
        start_alt = self.telemetry.get("surface_altitude", 0.0)
        while time.time() < deadline:
            self.control.autostage()
            alt = self.telemetry.get("surface_altitude", 0.0)
            if alt - start_alt > 15.0:
                self.rewards.award(Milestone.LIFTOFF, self.telemetry.get("mission_time"),
                                   {"twr": round(twr, 2)})
                self._fact("Ракета оторвалась от стартового стола",
                           f"TWR {twr:.2f}, высота {alt:.0f} м")
                log.info("Отрыв от стола! TWR=%.2f", twr)
                return True
            time.sleep(0.1)

        log.error("Ракета не оторвалась от стола за 20 с")
        self.rewards.penalise(Failure.NO_LIFTOFF, self.telemetry.get("mission_time"),
                              {"twr": round(twr, 2)})
        return False

    # ------------------------------------------------------------------
    def gravity_turn(self, timeout: float = 900.0) -> bool:
        """Ведёт ракету до достижения целевого апоапсиса."""
        self._status("Выполнение гравитационного манёвра",
                     f"разворот {self.g.turn_start_altitude:.0f}→"
                     f"{self.g.turn_end_altitude:.0f} м, цель Ap "
                     f"{self.target_apoapsis/1000:.0f} км")
        deadline = time.time() + timeout
        tick = CONFIG.control.physics_tick
        last_report = 0.0
        best_apoapsis = 0.0
        stalled_since = None

        while time.time() < deadline:
            snap = self.telemetry.snapshot()
            self.rewards.evaluate_telemetry(snap, self.target_apoapsis)
            if self.rewards.terminated:
                return False
            if self._destroyed(snap):
                return False

            self.control.autostage()
            pitch = self.airflow_leash(snap, self.target_pitch(snap.altitude))

            # ТВЕРДОТОПЛИВНУЮ СВЯЗКУ ГАЗОМ НЕ УДЕРЖАТЬ.
            #
            # Весь подъём построен на том, что тягу можно убрать, когда
            # апоапсис набран. У «CL-30 Hestia» под нами 492 тонны
            # твёрдого топлива: его нельзя ни задросселировать, ни
            # выключить — он горит до конца, что бы пилот ни командовал.
            #
            # Единственный руль, который в этот момент ещё работает, —
            # тангаж. Опускаем нос к горизонту: лишняя энергия уходит не
            # вверх, задирая апоапсис в пустоту, а вбок, в горизонтальную
            # скорость, которая для орбиты и нужна.
            if snap.apoapsis > self.target_apoapsis and self._solid_burning():
                pitch = min(pitch, 0.0)
                if not self._said_solid:
                    self._said_solid = True
                    log.info("Апоапсис набран, но связка твердотопливная — "
                             "гашу лишнюю тягу тангажом, а не газом")

            self.control.point(pitch, self.g.ascent_heading)
            self.control.set_throttle(self.throttle_for(snap))

            if snap.mission_time - last_report > 15.0:
                # Пишем и КОМАНДУ, и ФАКТ. Раньше в журнал шла только
                # команда, и по нему нельзя было отличить «автопилот сам
                # развернул аппарат» от «аппарат развернуло, а автопилот
                # следом». Разбор трёх разрушений подряд упёрся ровно в это.
                target = self.target_pitch(snap.altitude)
                log.info("h=%.0f м | Ap=%.0f м | v=%.0f м/с | ст.%d | "
                         "профиль %.0f° -> команда %.0f° | ФАКТ тангаж %.0f°, "
                         "угол атаки %.0f°, скольжение %.0f°, напор %.0f Па",
                         snap.altitude, snap.apoapsis, snap.speed, snap.stage,
                         target, pitch, snap.pitch or 0.0,
                         snap.angle_of_attack or 0.0,
                         getattr(snap, "sideslip_angle", 0.0) or 0.0,
                         snap.dynamic_pressure or 0.0)
                self._status("Вывод на орбиту (гравитационный разворот)",
                             f"h={snap.altitude:.0f} м, Ap={snap.apoapsis:.0f} м, "
                             f"v={snap.speed:.0f} м/с, тангаж {pitch:.0f}°")
                last_report = snap.mission_time

            # АПОАПСИС НЕ РАСТЁТ, А МЫ ПАДАЕМ — ПОДЪЁМ ОКОНЧЕН.
            #
            # Замер полёта к Дюне с опорной орбитой 300 км: топливо
            # ступени кончилось на 198.5 км, апоапсис замер намертво, а
            # цикл продолжал «вести» аппарат — тот прошёл вершину и
            # падал обратно: 144 км, 122, 97, 83, 67, и так до самой
            # атмосферы, где сгорел бы. Цикл ждал 300 км, которых уже
            # неоткуда было взять.
            #
            # Лучше скруглять то, что есть, чем везти аппарат в землю.
            if snap.apoapsis > best_apoapsis + 200.0:
                best_apoapsis = snap.apoapsis
                stalled_since = None
            elif snap.vertical_speed < -20.0 and snap.altitude > 50_000:
                stalled_since = stalled_since or time.time()
                if time.time() - stalled_since > 8.0:
                    self.control.set_throttle(0.0)
                    log.warning("Подъём окончен досрочно: апоапсис встал на "
                                "%.0f км при цели %.0f км, аппарат снижается "
                                "(%.0f м/с) — скругляем что есть",
                                snap.apoapsis / 1000.0,
                                self.target_apoapsis / 1000.0,
                                snap.vertical_speed)
                    self._fact(f"Подъём прерван: апоапсис {snap.apoapsis/1000:.0f} км",
                               "тяга кончилась, аппарат пошёл вниз")
                    return snap.apoapsis > 70_000.0

            if snap.apoapsis >= self.target_apoapsis and snap.altitude > 40_000:
                self.control.set_throttle(0.0)
                self._fact(f"Апоапсис {snap.apoapsis/1000:.1f} км набран",
                           f"двигатели выключены на высоте {snap.altitude:.0f} м")
                log.info("Целевой апоапсис достигнут: %.0f м", snap.apoapsis)
                return True

            if self._tumbling(snap):
                return False
            if self._ascent_hopeless(snap):
                return False

            if self._out_of_fuel(snap):
                self.rewards.penalise(Failure.OUT_OF_FUEL, snap.mission_time,
                                      {"altitude": round(snap.altitude),
                                       "apoapsis": round(snap.apoapsis)})
                return False
            time.sleep(tick)

        log.error("Таймаут выведения")
        self.rewards.penalise(Failure.TIMEOUT, self.telemetry.get("mission_time"))
        return False

    # ------------------------------------------------------------------
    def coast_to_space(self, timeout: float = 600.0) -> bool:
        """Выбег до выхода из атмосферы с поддержкой апоапсиса."""
        self._status("Выбег до границы атмосферы")
        deadline = time.time() + timeout
        body = self.telemetry.vessel.orbit.body
        atmo = body.atmosphere_depth if body.has_atmosphere else 0.0

        self.control.point(0.0, self.g.ascent_heading)
        while time.time() < deadline:
            snap = self.telemetry.snapshot()
            self.rewards.evaluate_telemetry(snap, self.target_apoapsis)
            if self.rewards.terminated or self._destroyed(snap):
                return False

            if snap.altitude > atmo:
                self.control.set_throttle(0.0)
                log.info("Атмосфера пройдена: h=%.0f м, Ap=%.0f м",
                         snap.altitude, snap.apoapsis)
                self._fact(f"Атмосфера пройдена на {snap.altitude/1000:.1f} км",
                           f"Ap={snap.apoapsis/1000:.1f} км")
                # ПОРЯДОК ВАЖЕН: сначала створки, потом всё остальное.
                #
                # Панели и антенна живут ПОД обтекателем. Раскрывать их в
                # закрытой оболочке бессмысленно: солнце к панелям не
                # проходит, заряд садится, и обесточенный аппарат теряет
                # управление — он просто вращается. Атмосфера позади,
                # держать створки больше незачем.
                if self.control.jettison_fairings():
                    self._fact("Створки обтекателя сброшены",
                               f"высота {snap.altitude/1000:.1f} км, "
                               f"атмосфера позади")
                self.control.deploy_solar_panels()
                # Антенна раскрывается ЗДЕСЬ, а не на столе: в воздухе её
                # ломает напор (см. `launch`). Без неё беспилотный
                # аппарат теряет управление вдали от Кербина.
                self.control.deploy_antennas()
                if self.control.run_science():
                    self.rewards.award(Milestone.SCIENCE_COLLECTED, snap.mission_time)
                return True

            if snap.apoapsis < self.target_apoapsis * 0.98:
                self.control.set_throttle(min(1.0, self.g.coast_throttle + 0.2))
            else:
                self.control.set_throttle(0.0)
            self.control.autostage()
            time.sleep(CONFIG.control.physics_tick * 2)

        self.rewards.penalise(Failure.TIMEOUT, self.telemetry.get("mission_time"))
        return False

    # ------------------------------------------------------------------
    def circularize(self) -> bool:
        log.info("Скругление орбиты в апоапсисе")
        self._status("Скругление орбиты в апоапсисе")
        ok = self.maneuver.circularize_at_apoapsis(lead=self.g.circularization_lead)
        snap = self.telemetry.snapshot()
        self.rewards.evaluate_telemetry(snap, self.target_apoapsis)
        self.rewards.shape_orbit_quality(snap.apoapsis, snap.periapsis, snap.mission_time)
        if Milestone.STABLE_ORBIT in self.rewards.achieved:
            self._fact(f"Выведен на орбиту {snap.periapsis/1000:.0f}×"
                       f"{snap.apoapsis/1000:.0f} км",
                       f"наклонение {snap.inclination*57.3:.1f}°")
            return True
        log.warning("Скругление неполное: Ap=%.0f Pe=%.0f", snap.apoapsis, snap.periapsis)
        if self.flight_log:
            self.flight_log.warning("Орбита не замкнута",
                                    f"Ap={snap.apoapsis:.0f} м, Pe={snap.periapsis:.0f} м")
        return False

    # ------------------------------------------------------------------
    def run(self) -> bool:
        if not self.launch():
            return False
        if not self.gravity_turn():
            return False
        if not self.coast_to_space():
            return False
        return self.circularize()

    # ------------------------------------------------------------------
    def _tumbling(self, snap) -> bool:
        """Аппарат потерял устойчивость и летит боком.

        Формулировка отказа в журнале содержит слово «опрокинуло» не для
        красоты: проектное бюро (`pro_modules/designer_pro.py`) вычитывает
        журналы полётов по списку TUMBLE_WORDS и по числу таких случаев
        решает, добавлять ли оперение и резать ли мидель. Без этой записи
        петля «полетел — опрокинуло — исправил конструкцию» разомкнута.
        """
        if not getattr(snap, "aero_angles_known", False):
            return False                    # kRPC не отдаёт углы — не гадаем
        if snap.dynamic_pressure < TUMBLE_MIN_Q:
            return False
        # Считаем ПОЛНЫЙ угол между носом и потоком, а не только в плоскости
        # тангажа. Аппарат, которого закрутило по курсу, имеет угол атаки
        # около нуля при рыскании в тридцать градусов — по одному тангажу
        # такой срыв не виден вовсе, а летит она уже боком.
        total = math.hypot(float(snap.angle_of_attack or 0.0),
                           float(getattr(snap, "sideslip_angle", 0.0) or 0.0))
        if total < TUMBLE_ANGLE:
            self._tumble_since = None
            return False

        if self._tumble_since is None:
            self._tumble_since = snap.mission_time
            return False
        if snap.mission_time - self._tumble_since < TUMBLE_HOLD:
            return False

        self.control.set_throttle(0.0)
        detail = {"angle_of_attack": round(snap.angle_of_attack, 1),
                  "sideslip_angle": round(float(getattr(snap, "sideslip_angle", 0.0) or 0.0), 1),
                  "total_angle": round(total, 1),
                  "altitude": round(snap.altitude),
                  "speed": round(snap.speed),
                  "dynamic_pressure": round(snap.dynamic_pressure)}
        self.rewards.penalise(Failure.LOST_CONTROL, snap.mission_time, detail)
        self._fact("Аппарат опрокинуло набегающим потоком",
                   f"нос отклонён от потока на {total:.0f}° и держится "
                   f"{TUMBLE_HOLD:.0f} с на высоте {snap.altitude:.0f} м, "
                   f"скорость {snap.speed:.0f} м/с — устойчивости не хватает, "
                   f"нужно оперение или меньший мидель")
        log.error("Потеря устойчивости: нос на %.0f° от потока (тангаж %.0f°, "
                  "рыскание %.0f°) на высоте %.0f м", total,
                  snap.angle_of_attack,
                  float(getattr(snap, "sideslip_angle", 0.0) or 0.0),
                  snap.altitude)
        return True

    def _ascent_hopeless(self, snap) -> bool:
        """Апоапсис падает при работающем двигателе — выведение сорвано.

        Ждать здесь нечего: аппарат уже не поднимается, а тормозится, и
        каждая лишняя секунда — это минуты живого времени, потраченные на
        полёт с известным исходом.
        """
        apoapsis = float(snap.apoapsis or 0.0)
        best = self._best_apoapsis
        if apoapsis > best:
            self._best_apoapsis = apoapsis
            self._apoapsis_falling_since = None
            return False
        # Считаем падением только заметное: мелкие колебания апоапсиса
        # бывают и на исправном выведении.
        if apoapsis > best * 0.97:
            return False
        if self._apoapsis_falling_since is None:
            self._apoapsis_falling_since = snap.mission_time
            return False
        if snap.mission_time - self._apoapsis_falling_since < APOAPSIS_DECAY_HOLD:
            return False

        self.control.set_throttle(0.0)
        # Тангаж в момент срыва отличает «летел слишком вертикально» от
        # «завалился слишком рано». Без него оба случая выглядят одинаково
        # («апоапсис упал»), и аналитик лечил их одним лекарством —
        # разворачивайся раньше, — загоняя профиль всё ниже.
        detail = {"apoapsis": round(apoapsis), "best_apoapsis": round(best),
                  "altitude": round(snap.altitude),
                  "pitch": round(float(getattr(snap, "pitch", 0.0) or 0.0), 1),
                  "target_pitch": round(self.target_pitch(snap.altitude), 1),
                  "vertical_speed": round(snap.vertical_speed, 1)}
        self.rewards.penalise(Failure.LOST_CONTROL, snap.mission_time, detail)
        self._fact("Выведение сорвано: апоапсис падает",
                   f"максимум был {best/1000:.1f} км, сейчас "
                   f"{apoapsis/1000:.1f} км, вертикальная скорость "
                   f"{snap.vertical_speed:+.0f} м/с — профиль тангажа не даёт "
                   f"набрать горизонтальную скорость")
        log.error("Выведение сорвано: апоапсис %.0f м против максимума %.0f м",
                  apoapsis, best)
        return True

    def _destroyed(self, snap) -> bool:
        try:
            if not self.connection.is_alive():
                self.rewards.penalise(Failure.CONNECTION_LOST, snap.mission_time)
                return True
        except Exception:
            self.rewards.penalise(Failure.CONNECTION_LOST, snap.mission_time)
            return True
        if snap.part_count == 0:
            self.rewards.penalise(Failure.EXPLOSION, snap.mission_time)
            if self.flight_log:
                self.flight_log.error("Зафиксировано разрушение аппарата")
            return True
        if snap.situation in ("landed", "splashed") and snap.mission_time > 20:
            self.rewards.penalise(Failure.CRASH, snap.mission_time,
                                  {"altitude": round(snap.altitude),
                                   "speed": round(snap.speed)})
            if self.flight_log:
                self.flight_log.error("Зафиксировано падение",
                                      f"скорость {snap.speed:.0f} м/с")
            return True
        return False

    @staticmethod
    def _out_of_fuel(snap) -> bool:
        lf = snap.resources.get("LiquidFuel", {})
        if lf.get("max", 0) <= 0:
            return False
        return lf.get("amount", 0.0) < 0.2 and snap.available_thrust < 1.0
