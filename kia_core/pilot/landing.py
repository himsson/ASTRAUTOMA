"""Посадка на безатмосферное тело: сход с орбиты и торможение у грунта.

Почему отдельным модулем. Подъём, манёвры и перелёт исполняются узлами:
там всё сводится к «повернуться и отработать импульс». Посадка так не
считается — тяга сравнима с весом, тело вращается под аппаратом, а
ошибка в сотню метров означает удар. Поэтому здесь обратная связь по
высоте и скорости, без единого манёвренного узла.

Порядок работы такой же, какой применил бы живой пилот:

1. СХОД С ОРБИТЫ. Тормозим в апоапсисе, пока перицентр не опустится под
   поверхность. Опускать сразу к нулю нельзя: аппарат придёт к грунту с
   огромной горизонтальной скоростью, и гасить её будет нечем.

2. ГАШЕНИЕ ГОРИЗОНТАЛЬНОЙ СКОРОСТИ. Держим ретроградный вектор
   ОТНОСИТЕЛЬНО ПОВЕРХНОСТИ и жжём, пока горизонтальная составляющая не
   станет пренебрежимой. На Муне это около 550 м/с с низкой орбиты.

3. ВЕРТИКАЛЬНЫЙ СПУСК. Дальше держим скорость снижения по расписанию:
   чем ниже, тем медленнее. Расписание линейное по высоте — простое и
   устойчивое, в отличие от «идеального» суицидального ожога, который
   разваливается от любой ошибки в тяге.

4. КАСАНИЕ. Опоры выпускаются заранее, на километре: их раскрытие
   занимает время, а на последних метрах его уже нет.
"""
from __future__ import annotations

import math
import time

from ..logging_setup import get_logger
from ..learning.rewards import Failure, Milestone

log = get_logger("pilot.landing")


class LandingPilot:
    """Сажает аппарат на безатмосферное тело по обратной связи."""

    # Высота перицентра, к которой сводим орбиту перед торможением.
    # Ноль ставить нельзя: аппарат должен успеть погасить горизонтальную
    # скорость ДО того, как окажется у грунта.
    # Замер живого спуска: при перицентре 8 км аппарат приходил к
    # торможению на 2.6 км, имея 745 м/с горизонтальной скорости — до
    # грунта секунды, гасить нечем. Держим выше и начинаем раньше.
    DEORBIT_PERIAPSIS = 15_000.0
    # До какой высоты просто падаем после схода с орбиты, м.
    COAST_UNTIL = 30_000.0
    # Горизонтальная скорость, ниже которой считаем её погашенной, м/с.
    # ВОСЕМЬ МЕТРОВ В СЕКУНДУ — ЭТО НЕ «ПОГАШЕНО».
    # Оператор увидел касание с горизонтальной 30 м/с: аппарат не садится,
    # а чиркает по грунту и опрокидывается. Опора держит удар вниз, но не
    # боковой снос. Два метра в секунду — шаг пешехода.
    HORIZONTAL_DONE = 2.0
    # Высота выпуска опор, м. Раскрытие занимает секунды.
    LEGS_ALTITUDE = 1_500.0
    # Расписание снижения: скорость = BASE + высота / SLOPE, м/с.
    # МЯГЧЕ У САМОГО ГРУНТА.
    #
    # Замер посадки: касание 28.1 м/с при том, что опора LT-2 держит
    # около двенадцати. Прежнее расписание на сотне метров разрешало
    # снижение 4 + 100/12 ≈ 12 м/с, и остаток гасить было уже нечем.
    # Теперь у земли график вдвое строже: 2 + h/25 даёт на сотне метров
    # шесть метров в секунду, на десяти — два с половиной.
    DESCENT_BASE = 2.0
    DESCENT_SLOPE = 25.0
    # Быстрее этого не снижаемся ни на какой высоте.
    DESCENT_MAX = 120.0
    # Считаем, что сели, если ситуация «на поверхности» или скорость
    # почти нулевая у самого грунта.
    TOUCHDOWN_SPEED = 1.0

    def __init__(self, connection, telemetry, control, maneuver, rewards,
                 target_name: str, flight_log=None):
        self.connection = connection
        self.telemetry = telemetry
        self.control = control
        self.maneuver = maneuver
        self.rewards = rewards
        self.target_name = target_name
        self.flight_log = flight_log
        self.sc = connection.space_center

    # ------------------------------------------------------------------
    @property
    def vessel(self):
        return self.telemetry.vessel

    def _status(self, text: str) -> None:
        log.info("СТАТУС: %s", text)
        if self.flight_log:
            self.flight_log.status(text)

    def _fact(self, text: str, detail: str = "") -> None:
        log.info("ФАКТ: %s%s", text, f" — {detail}" if detail else "")
        if self.flight_log:
            self.flight_log.fact(text, detail)

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Полный цикл посадки. True — аппарат стоит на поверхности."""
        vessel = self.vessel
        body = vessel.orbit.body
        if body.has_atmosphere:
            log.warning("Посадка в атмосфере не реализована — только вакуум")
            return False
        if not self.deorbit():
            return False
        if not self.coast_to_surface():
            return False
        if not self.kill_horizontal():
            return False
        return self.descend()

    # ------------------------------------------------------------------
    def coast_to_surface(self) -> bool:
        """Ждёт подхода к грунту после схода с орбиты.

        БЕЗ ЭТОГО ТОРМОЖЕНИЕ НАЧИНАЛОСЬ В АПОАПСИСЕ.
        Захват у Муны даёт высокую орбиту — в живом полёте это было
        1273 км. Опустив перицентр, аппарат остаётся на прежней высоте и
        должен ещё долететь до низа. Прежний код шёл гасить скорость
        сразу и жёг топливо в полутора тысячах километров от цели.
        """
        vessel = self.vessel
        frame = vessel.orbit.body.reference_frame
        start = vessel.flight(frame).surface_altitude
        if start <= self.COAST_UNTIL:
            return True
        self._status(f"Выбег к поверхности {self.target_name}")
        # Разворачиваемся заранее — на подлёте будет не до того.
        try:
            ap = self.control.ap
            ap.reference_frame = vessel.surface_velocity_reference_frame
            ap.target_direction = (0.0, -1.0, 0.0)
            self.control.engage_autopilot()
        except Exception:
            pass
        deadline = time.time() + 900.0
        while time.time() < deadline:
            altitude = vessel.flight(frame).surface_altitude
            if altitude <= self.COAST_UNTIL:
                log.info("Подошли к поверхности: h=%.1f км", altitude / 1000.0)
                return True
            # ВАРП КОРОТКИМИ ШАГАМИ, А НЕ ОДНИМ ПРЫЖКОМ.
            #
            # Прыжок «за 75 секунд до перицентра» проскакивал нужную
            # высоту: аппарат приходил в себя уже на 2.6 км, имея 745 м/с
            # горизонтальной скорости. Погасить её оттуда нечем — до
            # грунта секунды. Шагаем понемногу и после каждого шага
            # смотрим высоту.
            # ШАГ СЧИТАЕМ ПО СКОРОСТИ СНИЖЕНИЯ, А НЕ НАУГАД.
            #
            # Постоянные тридцать секунд оказались и слишком мелкими, и
            # слишком глупыми: с 374 км при снижении 210 м/с до цели
            # больше полутора тысяч секунд, то есть полсотни шагов. В
            # игре это выглядело как «варп включился и сразу выключился»,
            # и так десятками раз. Оценка простая: сколько осталось
            # падать, столько и warp, но с запасом на ошибку.
            drop = vessel.flight(frame).vertical_speed
            to_peri = vessel.orbit.time_to_periapsis
            if drop < -1.0:
                need = (altitude - self.COAST_UNTIL) / (-drop)
                step = max(5.0, min(need * 0.6, max(5.0, to_peri - 20.0)))
            else:
                step = max(5.0, min(60.0, to_peri * 0.3))
            self.control.warp_to(self.sc.ut + step)
            time.sleep(0.2)
        log.warning("Выбег к поверхности не завершился вовремя")
        return True

    # ------------------------------------------------------------------
    def deorbit(self) -> bool:
        """Опускает перицентр под поверхность, тормозя в апоапсисе."""
        vessel = self.vessel
        if vessel.orbit.periapsis_altitude <= self.DEORBIT_PERIAPSIS:
            log.info("Перицентр уже %.1f км — сход с орбиты не нужен",
                     vessel.orbit.periapsis_altitude / 1000.0)
            return True

        self._status(f"Сход с орбиты {self.target_name}")
        # Тормозим В АПОАПСИСЕ: там импульс дешевле всего опускает
        # противоположную сторону орбиты.
        wait = vessel.orbit.time_to_apoapsis
        if wait > 20.0:
            self.control.warp_to(self.sc.ut + wait - 15.0)

        ap = self.control.ap
        try:
            ap.reference_frame = vessel.orbital_reference_frame
            ap.target_direction = (0.0, -1.0, 0.0)      # ретроградно
            self.control.engage_autopilot()
        except Exception as exc:
            log.error("Ориентация на торможение не удалась: %s", exc)
            return False
        self.control.wait_for_orientation(tolerance=5.0, timeout=60.0)

        log.info("Сход с орбиты: перицентр %.1f км -> %.1f км",
                 vessel.orbit.periapsis_altitude / 1000.0,
                 self.DEORBIT_PERIAPSIS / 1000.0)
        deadline = time.time() + 240.0
        self.control.set_throttle(1.0)
        try:
            while time.time() < deadline:
                periapsis = vessel.orbit.periapsis_altitude
                if periapsis <= self.DEORBIT_PERIAPSIS:
                    break
                # У самой цели сбавляем газ, чтобы не провалиться слишком
                # глубоко: лишние сотни метров в секунду потом придётся
                # гасить у грунта, где каждый метр в секунду дорог.
                gap = periapsis - self.DEORBIT_PERIAPSIS
                self.control.set_throttle(1.0 if gap > 20_000.0
                                          else max(0.1, gap / 20_000.0))
                if self.control.should_stage():
                    self.control.autostage()
                time.sleep(0.2)
        finally:
            self.control.set_throttle(0.0)
        self._fact("Сход с орбиты выполнен",
                   f"перицентр {vessel.orbit.periapsis_altitude/1000:.1f} км")
        return True

    # ------------------------------------------------------------------
    def _prepare_touchdown(self) -> None:
        """Сбрасывает отработавшее и выпускает опоры перед касанием.

        Оператор перечислил ровно то, чего не было: последняя ступень не
        отделена и ножки не выдвинуты. Ступень с сухим баком висит внизу
        и принимает удар вместо опор, а сами опоры без команды остаются
        сложенными — раскрытие занимает несколько секунд, и давать её
        надо заранее.
        """
        if getattr(self, "_touchdown_ready", False):
            return
        try:
            if self.control.should_stage():
                self.control.autostage()
        except Exception:
            pass
        opened = self.control.deploy_landing_gear()
        try:
            total = len(self.telemetry.vessel.parts.legs)
        except Exception:
            total = 0
        log.info("Готовлюсь к касанию: опор раскрыто %d из %d", opened, total)
        self._fact("Опоры выпущены", f"{opened} из {total}")
        self._touchdown_ready = True

    def kill_horizontal(self) -> bool:
        """Гасит горизонтальную скорость, держа ретроград ОТНОСИТЕЛЬНО ГРУНТА.

        Орбитальный ретроград здесь не годится: у поверхности важна
        скорость относительно самой поверхности, а тело под аппаратом
        вращается.
        """
        vessel = self.vessel
        body = vessel.orbit.body
        frame = body.reference_frame
        self._status(f"Гашение горизонтальной скорости у {self.target_name}")

        ap = self.control.ap
        try:
            ap.reference_frame = vessel.surface_velocity_reference_frame
            ap.target_direction = (0.0, -1.0, 0.0)      # против скорости
            self.control.engage_autopilot()
        except Exception as exc:
            log.error("Ориентация против скорости не удалась: %s", exc)
            return False

        deadline = time.time() + 600.0
        last_report = 0.0
        while time.time() < deadline:
            flight = vessel.flight(frame)
            speed = flight.speed
            vertical = flight.vertical_speed
            horizontal = math.sqrt(max(0.0, speed * speed - vertical * vertical))
            altitude = flight.surface_altitude

            if altitude <= self.LEGS_ALTITUDE:
                self._prepare_touchdown()

            if horizontal <= self.HORIZONTAL_DONE:
                self.control.set_throttle(0.0)
                self._fact("Горизонтальная скорость погашена",
                           f"осталось {horizontal:.1f} м/с на высоте "
                           f"{altitude:.0f} м")
                return True

            # ГРУНТ ВАЖНЕЕ ГОРИЗОНТА. Если снижаемся быстро, а высоты
            # мало, гасить надо вертикаль, иначе разобьёмся с идеально
            # погашенным горизонтом.
            if altitude < 3_000.0 and vertical < -60.0:
                self.control.set_throttle(1.0)
            else:
                self.control.set_throttle(1.0 if horizontal > 40.0
                                          else max(0.15, horizontal / 40.0))

            if self.control.should_stage():
                self.control.autostage()
            if time.time() - last_report > 5.0:
                last_report = time.time()
                log.info("Торможение: h=%.0f м, гориз=%.0f м/с, верт=%.0f м/с",
                         altitude, horizontal, vertical)
            if altitude < 50.0:
                break
            time.sleep(0.2)

        self.control.set_throttle(0.0)
        log.warning("Горизонтальную скорость погасить не успели")
        return True        # пробуем сесть тем, что осталось

    # ------------------------------------------------------------------
    def descend(self) -> bool:
        """Вертикальный спуск по расписанию скорости и касание."""
        vessel = self.vessel
        body = vessel.orbit.body
        frame = body.reference_frame
        self._status(f"Спуск на поверхность {self.target_name}")

        ap = self.control.ap
        try:
            # Носом вверх по местной вертикали: тяга смотрит вниз, в грунт.
            ap.reference_frame = vessel.surface_reference_frame
            ap.target_direction = (1.0, 0.0, 0.0)       # «вверх» в этой системе
            self.control.engage_autopilot()
        except Exception as exc:
            log.error("Ориентация на спуск не удалась: %s", exc)
            return False

        deadline = time.time() + 900.0
        last_report = 0.0
        while time.time() < deadline:
            flight = vessel.flight(frame)
            altitude = flight.surface_altitude
            vertical = flight.vertical_speed
            speed = flight.speed

            if altitude <= self.LEGS_ALTITUDE:
                self._prepare_touchdown()

            situation = str(vessel.situation)
            if "landed" in situation or "splashed" in situation:
                self.control.set_throttle(0.0)
                snap = self.telemetry.snapshot()
                self.rewards.evaluate_telemetry(snap)
                self._fact(f"ПОСАДКА на {self.target_name} выполнена",
                           f"скорость касания {speed:.1f} м/с")
                log.info("ПОСАДКА: %s, скорость %.1f м/с", self.target_name, speed)
                return True

            if altitude < 3.0 and abs(speed) < self.TOUCHDOWN_SPEED:
                self.control.set_throttle(0.0)
                self._fact(f"Касание грунта {self.target_name}",
                           f"скорость {speed:.1f} м/с")
                return True

            # Расписание: чем ниже, тем медленнее падаем.
            target = -min(self.DESCENT_MAX,
                          self.DESCENT_BASE + altitude / self.DESCENT_SLOPE)
            # ЗНАК ЗДЕСЬ БЫЛ ПЕРЕПУТАН, И ЭТО СТОИЛО ПОЛЁТА.
            #
            # Тормозить надо, только когда падаешь БЫСТРЕЕ расписания.
            # В первой версии условие было обратным, и аппарат жёг на
            # полной тяге вверх, снижаясь медленнее цели:
            #
            #     цель −120.0 м/с, факт −0.3 м/с, газ 1.00
            #     h=1470 км -> 1693 км, верт +1744 м/с
            #
            # То есть он улетал от Муны, а журнал называл это спуском.
            if vertical < target:              # падаем быстрее расписания
                # ОТСТАВАНИЕ ОТ ГРАФИКА ОТРАБАТЫВАЕМ РЕЗЧЕ.
                # Делитель 12 означал «полная тяга только при отставании
                # на 12 м/с» — у грунта такая мягкость и дала удар в 28
                # м/с. Тройка снимает остаток заранее.
                throttle = min(1.0, (target - vertical) / 3.0)
            else:
                throttle = 0.0                 # тяготение само разгонит
            self.control.set_throttle(throttle)

            if self.control.should_stage():
                self.control.autostage()
            if time.time() - last_report > 5.0:
                last_report = time.time()
                log.info("Спуск: h=%.0f м, верт=%.1f м/с (цель %.1f), газ %.2f",
                         altitude, vertical, target, throttle)
            time.sleep(0.1)

        self.control.set_throttle(0.0)
        log.warning("Спуск не завершился за отведённое время")
        return False
