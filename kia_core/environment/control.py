"""«Руки» ракеты: тяга, ступени, ориентация, SAS, варп, запуск судна."""
from __future__ import annotations

import math
import time

from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("env.control")


class PID:
    """Классический ПИД с ограничением выхода и антивиндапом."""

    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = -1.0, out_max: float = 1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def update(self, error: float, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        if self._last_time is None:
            dt = CONFIG.control.physics_tick
        else:
            dt = max(1e-3, now - self._last_time)

        self._integral += error * dt
        # антивиндап: держим интеграл в пределах, дающих допустимый вклад
        if self.ki > 0:
            limit = (self.out_max - self.out_min) / self.ki
            self._integral = max(-limit, min(limit, self._integral))

        derivative = 0.0 if self._last_error is None else (error - self._last_error) / dt
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._last_error, self._last_time = error, now
        return max(self.out_min, min(self.out_max, output))


class ShipControl:
    """Прямое управление активным судном через kRPC."""

    def __init__(self, connection, telemetry=None, flight_log=None):
        self.connection = connection
        self.sc = connection.space_center
        self.telemetry = telemetry
        self.flight_log = flight_log
        self.vessel = telemetry.vessel if telemetry else self.sc.active_vessel

    def rebind(self) -> None:
        self.vessel = self.sc.active_vessel

    # ------------------------------------------------------------------
    # Тяга и ступени
    # ------------------------------------------------------------------
    @property
    def throttle(self) -> float:
        return self.vessel.control.throttle

    def set_throttle(self, value: float) -> None:
        self.ctl.throttle = max(0.0, min(1.0, value))

    def ignite(self, throttle: float = 1.0) -> None:
        """Зажигание: полная тяга + активация первой ступени."""
        self.set_throttle(throttle)
        stage = self.vessel.control.current_stage
        self.vessel.control.activate_next_stage()
        log.info("Зажигание! Тяга=%.0f%%", throttle * 100)
        if self.flight_log:
            self.flight_log.fact("Запуск двигателей первой ступени",
                                 f"стадия {stage}, тяга {throttle*100:.0f}%")

    def next_stage(self) -> int:
        stage = self.vessel.control.current_stage
        self.vessel.control.activate_next_stage()
        new_stage = self.vessel.control.current_stage
        log.info("Отделение ступени %d -> %d", stage, new_stage)
        if self.flight_log:
            thrust = 0.0
            try:
                thrust = self.vessel.available_thrust / 1000.0
            except Exception:
                pass
            self.flight_log.fact(f"Отделена ступень (стадия {stage} → {new_stage})",
                                 f"доступная тяга после разделения {thrust:.0f} кН")
        return new_stage

    def should_stage(self) -> bool:
        """Пора ли отстреливать: работающие двигатели выдохлись.

        СПРАШИВАЕМ ДВИГАТЕЛИ, А НЕ ГРУППЫ РАЗДЕЛЕНИЯ.
        Прежний расчёт смотрел остаток топлива в группе, которая
        отделяется следующей. С обычной «сосиской» это работало, но с
        боковыми ускорителями группа содержит и твердотопливные шашки, и
        жидкое топливо центрального блока сразу. Шашки выгорают первыми,
        условие срабатывает — и ракета делится дальше, хотя центральный
        двигатель ещё тянет. Отсюда и запись в журнале «Отделение
        ступени 0 -> 0» через тринадцать секунд после отрыва, и четыре
        провала из четырёх, из-за которых ускорители были запрещены.
        Физический признак надёжнее: если ХОТЬ ОДИН работающий двигатель
        ещё даёт тягу, делить нечего.

        ПРОВЕРКА НЕ ЧАЩЕ ДВУХ РАЗ В СЕКУНДУ. Опрос каждого двигателя —
        отдельный запрос к игре, а цикл выведения зовёт её 20 раз в
        секунду: на ракете с двадцатью двигателями это сотни запросов в
        секунду и падение FPS с 160 до 5. Полсекунды задержки отделения
        на полёт не влияют.
        """
        now = time.monotonic()
        if now - getattr(self, "_stage_checked", 0.0) < 0.5:
            return False
        self._stage_checked = now
        try:
            active = [e for e in self.vessel.parts.engines if e.active]
            if active and self.vessel.control.throttle > 0.05:
                # СБРАСЫВАЕМ, КАК ТОЛЬКО ХОТЬ ОДИН АКТИВНЫЙ ВЫГОРЕЛ.
                #
                # Прежнее правило звучало «пока хоть один работающий даёт
                # тягу — делить нечего», и это была ошибка в обратную
                # сторону. Замер живого полёта: связка «Колотушек» горит
                # 44 с, а центральный «Бобкэт» — минуты. Пустые
                # ускорители, тридцать тонн, висели на борту ПЯТЬ МИНУТ:
                #
                #     22:13:11 Отрыв, TWR=2.28
                #     22:13:39 ... ст.6
                #     22:17:39 ... ст.6   <- та же стадия
                #     22:18:00 Отделение ступени 6 -> 5
                #
                # Апоапсис застрял на 50 км, ракета не вышла на орбиту.
                # Признак выгорания — `has_fuel`: он не зависит от
                # раскрутки двигателя и не врёт в первые секунды.
                if any(not e.has_fuel for e in active):
                    return True
                return False
        except Exception as exc:
            log.debug("should_stage по двигателям: %s", exc)
        try:
            stage = self.vessel.control.current_stage
            resources = self.vessel.resources_in_decouple_stage(stage=stage - 1,
                                                                cumulative=False)
            for res in ("LiquidFuel", "Oxidizer", "SolidFuel"):
                if res in resources.names and resources.max(res) > 0:
                    if resources.amount(res) < 0.1:
                        return True
        except Exception as exc:
            log.debug("should_stage: %s", exc)
        # запасной критерий — нулевая доступная тяга при открытом дросселе
        try:
            if (self.vessel.control.throttle > 0.05
                    and self.vessel.available_thrust < 1.0
                    and self.vessel.control.current_stage > 0):
                return True
        except Exception:
            pass
        return False

    def autostage(self, min_stage: int = 0) -> bool:
        """Отстреливает ОДНУ пустую ступень за вызов. True — отстрелена.

        Здесь стоял цикл `while`, и это переворачивало ракету.
        Сразу после отделения новая ступень ещё не ожила: двигатель не
        вышел на режим, а запрос остатков по группе разделения отдаёт
        пустоту. Условие `should_stage` снова истинно, и цикл делит
        второй раз, третий — пока не дойдёт до конца.

        Как это выглядело в игре:

            h=16987 м   тангаж  +67°, угол атаки -4°   идёт ровно
            Отделение ступени 2 -> 1   тяга после 20 кН
            Отделение ступени 1 -> 0   тяга после 20 кН
            h=26754 м   тангаж  -51°                    перевернуло

        Сто восемнадцать градусов за две секунды: аппарат остался без
        тяги посреди разворота, и поток довернул его сам.

        Одно отделение за вызов проблему снимает: цикл выведения зовёт
        автостейджинг каждый тик, и по-настоящему пустая ступень уйдёт
        на следующем — но уже по СВЕЖЕМУ замеру, а не по устаревшему.
        """
        if not self.should_stage():
            return False
        if self.vessel.control.current_stage <= min_stage:
            return False

        throttle = self.vessel.control.throttle
        self.set_throttle(0.0)
        time.sleep(0.25)
        self.next_stage()
        # Даём новой ступени ожить: раскрутка насосов и выход двигателя на
        # режим занимают около секунды, и всё это время игра честно
        # показывает нулевую тягу.
        time.sleep(1.0)
        self.set_throttle(throttle)
        time.sleep(0.4)
        return True

    # ------------------------------------------------------------------
    # Ориентация
    # ------------------------------------------------------------------
    @property
    def ap(self):
        """Автопилот судна — один объект на судно, без запроса каждый такт."""
        cached = self.__dict__.get("_ap")
        if cached is None or cached[0] is not self.vessel:
            cached = (self.vessel, self.vessel.auto_pilot)
            self._ap = cached
        return cached[1]

    @property
    def ctl(self):
        """Панель управления судна — тоже один объект на судно."""
        cached = self.__dict__.get("_ctl")
        if cached is None or cached[0] is not self.vessel:
            cached = (self.vessel, self.vessel.control)
            self._ctl = cached
        return cached[1]

    # МЯГКАЯ НАСТРОЙКА АВТОПИЛОТА kRPC.
    #
    # Со стандартными настройками (разгон 0.5 с, торможение 5 с) тяжёлая
    # ракета раскачивалась влево-вправо: регулятор разгонял поворот,
    # не успевал затормозить огромную инерцию, проскакивал цель и
    # возвращал обратно — и так весь подъём. Дольше торможение и
    # плавнее выход на пик — аппарат подходит к курсу без перелёта.
    SMOOTH_STOPPING_TIME = (1.0, 1.0, 1.0)      # с, запас на торможение вращения
    SMOOTH_DECELERATION_TIME = (10.0, 10.0, 10.0)  # с, плавный подход к цели
    SMOOTH_TIME_TO_PEAK = (4.0, 4.0, 4.0)       # с, без рывка при повороте
    SMOOTH_OVERSHOOT = (0.005, 0.005, 0.005)
    # Максимальная скорость изменения КОМАНДЫ тангажа, °/с: защита от
    # срыва потока иногда переключала команду скачком, и ракета дёргалась.
    PITCH_RATE_LIMIT = 3.0

    def tune_autopilot(self) -> None:
        ap = self.ap
        if getattr(self, "_tuned_ap", None) is ap:
            return
        for attr, value in (("auto_tune", True),
                            ("stopping_time", self.SMOOTH_STOPPING_TIME),
                            ("deceleration_time", self.SMOOTH_DECELERATION_TIME),
                            ("time_to_peak", self.SMOOTH_TIME_TO_PEAK),
                            ("overshoot", self.SMOOTH_OVERSHOOT)):
            try:
                setattr(ap, attr, value)
            except Exception as exc:
                log.debug("Автопилот: %s не задан (%s)", attr, exc)
        self._tuned_ap = ap

    def engage_autopilot(self, reference_frame=None) -> None:
        if reference_frame is not None:
            self.ap.reference_frame = reference_frame
        self.tune_autopilot()
        self._set_autopilot(True)

    def disengage_autopilot(self, hold: bool = True) -> None:
        """Отпускает автопилот, но по умолчанию оставляет удержание.

        БЕЗ УДЕРЖАНИЯ АППАРАТ КРУТИТСЯ ВЕСЬ ВЫБЕГ.
        Замер живого полёта: маховики включены, момент 48.9/30.5/48.9
        кН·м, заряд полный — а вращение стоит на 57.4 °/с и не гаснет,
        потому что команд на рули никто не подаёт. К ожогу автопилот
        выправляет аппарат за секунды (ошибка падает до 1°), но всё
        время между манёврами он кувыркается. Оператор видел это трижды
        и трижды сообщал.

        Штатный SAS в режиме удержания курса стоит дёшево: он гасит
        вращение маховиками, не трогая топливо.
        """
        try:
            self._set_autopilot(False)
        except Exception:
            pass
        if hold:
            self.set_sas(True, "stability_assist")

    def _set_autopilot(self, enabled: bool) -> None:
        """Разные сборки kRPC дают либо engage()/disengage(), либо свойство engaged."""
        ap = self.ap
        method = getattr(ap, "engage" if enabled else "disengage", None)
        if callable(method):
            method()
            return
        ap.engaged = enabled

    def point(self, pitch: float, heading: float, roll: float | None = None) -> None:
        """Направить нос: pitch — от горизонта (90 = вверх), heading — азимут.

        Команда тангажа меняется не быстрее PITCH_RATE_LIMIT °/с: скачок
        команды регулятор отрабатывает рывком, а рывок раскачивает ракету.
        """
        now = time.monotonic()
        last = getattr(self, "_last_point", None)
        if last is not None and now - last[0] < 2.0:
            step = self.PITCH_RATE_LIMIT * max(now - last[0], 0.02)
            pitch = max(last[1] - step, min(last[1] + step, pitch))
        self._last_point = (now, pitch)
        self.ap.target_pitch_and_heading(pitch, heading)
        if roll is not None:
            self.ap.target_roll = roll

    def point_at_vector(self, direction, reference_frame=None) -> None:
        if reference_frame is not None:
            self.ap.reference_frame = reference_frame
        self.ap.target_direction = direction

    def wait_for_orientation(self, tolerance: float = 5.0, timeout: float = 60.0) -> bool:
        """Ждёт, пока ошибка наведения не станет меньше tolerance градусов."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.ap.error < tolerance:
                    return True
            except Exception:
                return False
            time.sleep(0.1)
        log.warning("Наведение не сошлось за %.0f с (ошибка %.1f°)",
                    timeout, self._safe_error())
        return False

    def _safe_error(self) -> float:
        try:
            return self.ap.error
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # SAS / RCS / прочее
    # ------------------------------------------------------------------
    def set_sas(self, enabled: bool, mode: str | None = None) -> None:
        self.vessel.control.sas = enabled
        if enabled and mode:
            time.sleep(0.1)
            try:
                self.vessel.control.sas_mode = getattr(self.sc.SASMode, mode)
            except Exception as exc:
                log.debug("SAS-режим '%s' недоступен: %s", mode, exc)

    def set_rcs(self, enabled: bool) -> None:
        self.vessel.control.rcs = enabled

    def set_axes(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        """Ручное управление осями (когда автопилот отключён)."""
        c = self.vessel.control
        c.pitch, c.yaw, c.roll = pitch, yaw, roll

    def deploy_parachutes(self) -> None:
        count = 0
        for p in self.vessel.parts.parachutes:
            try:
                p.deploy()
                count += 1
            except Exception:
                pass
        if count:
            log.info("Раскрыто парашютов: %d", count)

    def deploy_landing_gear(self) -> int:
        """Выпускает посадочные опоры. Повторный вызов безвреден.

        Раскрытие занимает несколько секунд, поэтому команда даётся
        заранее — на полутора километрах, а не у самого грунта.
        """
        done = 0
        for leg in self.vessel.parts.legs:
            try:
                if not leg.deployed:
                    leg.deployed = True
                    done += 1
            except Exception:
                continue
        if done:
            log.info("Посадочные опоры выпущены: %d", done)
        return done

    def deploy_solar_panels(self) -> None:
        for p in self.vessel.parts.solar_panels:
            try:
                p.deployed = True
            except Exception:
                pass

    def jettison_fairings(self) -> int:
        """Раскрывает створки защитного обтекателя. Возвращает их число.

        БЕЗ ЭТОГО ВСЁ ПОД СТВОРКАМИ МЁРТВО.
        Сброса не было вовсе: обтекателю назначалась стадия, и он ждал
        автостейджинга, который случается по выгоранию двигателя, а не по
        выходу из атмосферы. Пока створки закрыты, солнечные панели под
        ними не видят солнца — заряд садится, и обесточенный зонд
        перестаёт слушаться руля. Оператор описал это тремя словами:
        «обтекатель не сброшен, панели не включены, ракета крутится» —
        и это была одна и та же беда.
        """
        done = 0
        for part in self.vessel.parts.all:
            try:
                fairing = part.fairing
                if fairing is not None and not fairing.jettisoned:
                    fairing.jettison()
                    done += 1
            except Exception:
                continue
        if done:
            log.info("Створки обтекателя сброшены: %d", done)
        return done

    def deploy_antennas(self) -> None:
        for a in self.vessel.parts.antennas:
            try:
                a.deployed = True
            except Exception:
                pass

    def run_science(self) -> int:
        """Запускает все доступные научные эксперименты."""
        done = 0
        for exp in self.vessel.parts.experiments:
            try:
                if exp.available and not exp.has_data:
                    exp.run()
                    done += 1
            except Exception:
                pass
        if done:
            log.info("Проведено экспериментов: %d", done)
        return done

    # ------------------------------------------------------------------
    # Время
    # ------------------------------------------------------------------
    def warp_to(self, ut: float, max_rate: float = 100_000.0) -> None:
        # У гиперболы нет апоапсиса, и расчёт момента даёт бесконечность.
        # Один такой полёт уже ушёл в «Варп на inf с вперёд» — дальше
        # игра не отвечала.
        if not math.isfinite(ut):
            log.warning("Варп отменён: момент времени не определён")
            return
        if ut <= self.sc.ut:
            return
        log.info("Варп на %.0f с вперёд", ut - self.sc.ut)
        try:
            self.sc.warp_to(ut, max_rails_rate=max_rate)
        except Exception as exc:
            log.warning("Варп не удался: %s", exc)

    def set_physics_warp(self, rate: int) -> None:
        try:
            self.sc.physics_warp_factor = max(0, min(CONFIG.control.max_time_warp, rate))
        except Exception as exc:
            log.debug("Физический варп недоступен: %s", exc)

    def stop_warp(self) -> None:
        try:
            self.sc.rails_warp_factor = 0
            self.sc.physics_warp_factor = 0
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Управление судном/сценой
    # ------------------------------------------------------------------
    def launch_craft(self, craft_name: str, launch_site: str = "LaunchPad",
                     directory: str = "VAB", recover: bool = True) -> bool:
        """Загружает .craft из VAB на стартовый стол.

        Сигнатура launch_vessel в разных сборках kRPC отличается: где-то
        crew — необязательный параметр, где-то обязательный позиционный.
        Пустой список означает «экипаж по умолчанию».
        """
        attempts = (
            lambda: self.sc.launch_vessel(directory, craft_name, launch_site, [],
                                          recover),
            lambda: self.sc.launch_vessel(directory, craft_name, launch_site,
                                          recover=recover),
            lambda: self.sc.launch_vessel_from_vab(craft_name, recover),
        )
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                attempt()
                log.info("Судно '%s' выведено на стартовый стол", craft_name)
                time.sleep(2.0)
                self.rebind()
                return True
            except TypeError as exc:
                last_error = exc            # не та сигнатура — пробуем следующую
                continue
            except Exception as exc:
                last_error = exc
                break
        log.error("Запуск '%s' не удался: %s", craft_name,
                  str(last_error).split("Server stack")[0].strip())
        return False

    def revert_to_launch(self) -> bool:
        try:
            self.sc.revert_to_launch()
            log.info("Откат к моменту старта")
            time.sleep(2.0)
            self.rebind()
            return True
        except Exception as exc:
            log.warning("Откат невозможен: %s", exc)
            return False

    def quicksave(self, name: str = "kia_autosave") -> None:
        try:
            self.sc.save(name)
        except Exception as exc:
            log.debug("Быстрое сохранение не удалось: %s", exc)

    def quickload(self, name: str = "kia_autosave") -> bool:
        try:
            self.sc.load(name)
            time.sleep(2.0)
            self.rebind()
            return True
        except Exception as exc:
            log.warning("Загрузка '%s' не удалась: %s", name, exc)
            return False

    def set_target_body(self, body_name: str) -> bool:
        try:
            self.sc.target_body = self.sc.bodies[body_name]
            log.info("Цель установлена: %s", body_name)
            return True
        except Exception as exc:
            log.warning("Не удалось задать цель %s: %s", body_name, exc)
            return False

    def clear_target(self) -> None:
        try:
            self.sc.clear_target()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def full_stop(self) -> None:
        """Аварийное «отпустить всё»."""
        try:
            self.set_throttle(0.0)
            self.disengage_autopilot()
            self.set_sas(False)
            self.stop_warp()
        except Exception:
            pass
