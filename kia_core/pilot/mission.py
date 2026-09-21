"""Миссия как конечный автомат: от стартового стола до цели, заданной оператором.

Куда лететь и на какую орбиту — берётся из MissionSpec, а не из констант.
Каждая фаза пишется в журнал полёта реального времени (kia_flight_log.md).
"""
from __future__ import annotations

import time
import traceback
from enum import Enum

from ..config import CONFIG
from ..environment.control import ShipControl
from ..environment.telemetry import Telemetry
from ..flight_log import FlightLog
from ..learning.rewards import Failure, Milestone, RewardSystem
from ..logging_setup import get_logger
from ..mission_spec import MissionSpec
from .ascent import AscentAutopilot
from .maneuver import ManeuverExecutor
from .transfer import TransferPlanner

log = get_logger("pilot.mission")


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    ASCENT = "ascent"
    COAST = "coast"
    CIRCULARIZE = "circularize"
    TRANSFER_PLAN = "transfer_plan"
    TRANSFER_BURN = "transfer_burn"
    COAST_TO_SOI = "coast_to_soi"
    CAPTURE = "capture"
    LANDING = "landing"
    COMPLETE = "complete"
    ABORTED = "aborted"


class Mission:
    """Одна попытка полёта. Экземпляр одноразовый."""

    def __init__(self, connection, flight_genome, spec: MissionSpec | None = None,
                 rewards: RewardSystem | None = None, flight_log: FlightLog | None = None,
                 abort_flag=None):
        self.connection = connection
        self.g = flight_genome
        self.spec = spec or MissionSpec.load()
        self.rewards = rewards or RewardSystem(spec=self.spec, flight_log=flight_log)
        self.flight_log = flight_log
        self.abort_flag = abort_flag          # threading.Event для команды «стоп»
        self.phase = Phase.PREFLIGHT
        self.started_wall = time.time()
        self.telemetry: Telemetry | None = None
        self.control: ShipControl | None = None
        self.maneuver: ManeuverExecutor | None = None
        self.final_snapshot = None
        self.phase_log: list[dict] = []

    # ------------------------------------------------------------------
    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        elapsed = time.time() - self.started_wall
        self.phase_log.append({"phase": phase.value, "wall_time": round(elapsed, 1),
                               "score": round(self.rewards.total, 1)})
        log.info("=== ФАЗА: %s (t+%.0f с, счёт %.0f) ===",
                 phase.value.upper(), elapsed, self.rewards.total)

    def _aborted(self) -> bool:
        if self.abort_flag is not None and self.abort_flag.is_set():
            log.warning("Получена команда прерывания полёта")
            if self.flight_log:
                self.flight_log.warning("Полёт прерван командой оператора")
            self.rewards.penalise(Failure.UNEXPECTED_ERROR, 0.0, {"reason": "оператор"})
            return True
        return False

    def _wall_timeout(self) -> bool:
        limit = self.spec.max_wall_time or CONFIG.mission.max_wall_time
        if time.time() - self.started_wall > limit:
            log.error("Превышен лимит реального времени на попытку")
            self.rewards.penalise(Failure.TIMEOUT,
                                  self.telemetry.get("mission_time") if self.telemetry else 0.0)
            return True
        return False

    def _check(self) -> bool:
        """True — можно продолжать."""
        return not (self._aborted() or self._wall_timeout())

    # ------------------------------------------------------------------
    def setup(self) -> bool:
        try:
            self.telemetry = Telemetry(self.connection)
            self.control = ShipControl(self.connection, self.telemetry,
                                       flight_log=self.flight_log)
            self.maneuver = ManeuverExecutor(self.connection, self.telemetry, self.control)
        except Exception as exc:
            log.error("Не удалось подготовить телеметрию/управление: %s", exc)
            self.rewards.penalise(Failure.UNEXPECTED_ERROR, 0.0, {"error": str(exc)})
            return False

        if self.flight_log:
            self.flight_log.bind_clock(
                lambda: (self.telemetry.get("ut"), self.telemetry.get("mission_time")))

        snap = self.telemetry.snapshot()
        log.info("Предполётная проверка: '%s', масса %.2f т, TWR=%.2f, ступеней %d",
                 self.telemetry.vessel.name, snap.mass, snap.twr, snap.stage)
        if self.flight_log:
            self.flight_log.fact(
                f"Аппарат на стартовом столе: {self.telemetry.vessel.name}",
                f"масса {snap.mass:.2f} т, деталей {snap.part_count}, "
                f"ступеней {snap.stage}, TWR {snap.twr:.2f}")
        if snap.situation not in ("pre_launch", "landed", "splashed"):
            log.warning("Судно уже в полёте (%s) — продолжаем с текущего состояния",
                        snap.situation)
        return True

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Выполняет миссию целиком, возвращает сводку."""
        try:
            if not self.setup():
                return self._finish(Phase.ABORTED)

            ascent = AscentAutopilot(
                self.connection, self.telemetry, self.control, self.maneuver,
                self.g, self.rewards, target_apoapsis=self.spec.park_orbit_altitude,
                flight_log=self.flight_log)

            self._enter(Phase.ASCENT)
            if not ascent.launch() or not self._check():
                return self._finish(Phase.ABORTED)
            if not ascent.gravity_turn() or not self._check():
                return self._finish(Phase.ABORTED)

            self._enter(Phase.COAST)
            if not ascent.coast_to_space() or not self._check():
                return self._finish(Phase.ABORTED)

            self._enter(Phase.CIRCULARIZE)
            if not ascent.circularize():
                log.warning("Стабильная орбита не достигнута — миссия прервана")
                self._penalise_wasted_fuel()
                return self._finish(Phase.ABORTED)
            if not self._check():
                return self._finish(Phase.ABORTED)

            # СПУТНИК СВЯЗИ СТАВИТСЯ В СВОЮ ТОЧКУ, А НЕ «НА ОРБИТУ».
            # Четыре аппарата вразнобой связи не дают: они собьются в
            # кучу и половина шара останется без покрытия.
            if self.spec.is_relay and not self.spec.needs_transfer:
                self._place_in_slot()

            # Задача «только орбита» — на этом всё
            if not self.spec.needs_transfer:
                snap = self.telemetry.snapshot()
                if snap.crew_count > 0:
                    self.rewards.award(Milestone.CREW_SURVIVED, snap.mission_time)
                return self._finish(Phase.COMPLETE)

            transfer = TransferPlanner(
                self.connection, self.telemetry, self.control, self.maneuver,
                self.g, self.rewards, target=self.spec.target_body,
                target_periapsis=self.spec.target_orbit_altitude,
                flight_log=self.flight_log)

            self._enter(Phase.TRANSFER_PLAN)
            node = transfer.plan_transfer()
            if node is None or not self._check():
                self._penalise_wasted_fuel()
                return self._finish(Phase.ABORTED)

            self._enter(Phase.TRANSFER_BURN)
            if not transfer.execute_transfer(node) or not self._check():
                self._penalise_wasted_fuel()
                return self._finish(Phase.ABORTED)

            self._enter(Phase.COAST_TO_SOI)
            # К ПЛАНЕТЕ ИДЁМ СТУПЕНЯМИ, К ЛУНЕ — НАПРЯМУЮ.
            # Разница в сроках: до Муны часы, до Дюны 302 суток, и на
            # таком плече курс надо править по дороге.
            cruise = (transfer.cruise_to_planet
                      if transfer._is_interplanetary() else transfer.coast_to_soi)
            if not cruise() or not self._check():
                self._penalise_wasted_fuel()
                return self._finish(Phase.ABORTED)

            self._enter(Phase.CAPTURE)
            captured = transfer.capture()

            # ПОСАДКА. Раньше миссия здесь заканчивалась: модуля спуска не
            # существовало вовсе, и аппарат, выйдя на орбиту цели, честно
            # отмечал достижение и завершал полёт. Оператор спросил прямо:
            # «почему он сделал идеальную орбиту и завершил полёт, а не
            # сел?» — потому что садиться было нечем.
            if captured and self.spec.is_relay:
                self._place_in_slot()

            if captured and self.spec.needs_landing:
                self._enter(Phase.LANDING)
                from .landing import LandingPilot
                lander = LandingPilot(
                    self.connection, self.telemetry, self.control,
                    self.maneuver, self.rewards,
                    target_name=self.spec.target_body,
                    flight_log=self.flight_log)
                lander.run()

            snap = self.telemetry.snapshot()
            if snap.crew_count > 0:
                self.rewards.award(Milestone.CREW_SURVIVED, snap.mission_time)
            return self._finish(Phase.COMPLETE)

        except KeyboardInterrupt:
            log.warning("Прервано пользователем")
            self.rewards.penalise(Failure.UNEXPECTED_ERROR, 0.0,
                                  {"error": "KeyboardInterrupt"})
            return self._finish(Phase.ABORTED)
        except Exception as exc:
            log.error("Сбой миссии: %s\n%s", exc, traceback.format_exc())
            failure = (Failure.CONNECTION_LOST if not self.connection.is_alive()
                       else Failure.UNEXPECTED_ERROR)
            self.rewards.penalise(failure, 0.0, {"error": str(exc)})
            if self.flight_log:
                self.flight_log.error("Сбой в ходе миссии", str(exc)[:200])
            return self._finish(Phase.ABORTED)

    # ------------------------------------------------------------------
    def _place_in_slot(self) -> bool:
        """Ставит спутник связи в назначенную точку созвездия."""
        phase = self.spec.constellation_phase
        try:
            from .constellation import ConstellationPilot
            pilot = ConstellationPilot(self.connection, self.control,
                                       self.maneuver, self.telemetry,
                                       self.flight_log)
            log.info("Точка созвездия %d из %d: %.0f°",
                     self.spec.constellation_slot + 1,
                     self.spec.constellation_size, phase)
            ok = pilot.place(phase)
        except Exception as exc:
            # Развод по фазе — доводка, а не условие полёта: спутник уже
            # на орбите и связь даёт. Ронять из-за неё миссию незачем.
            log.warning("Развод по фазе не удался: %s", exc)
            return False
        # У ЖУРНАЛА ПОЛЁТА НЕТ `info` — И ЭТО СТОИЛО ГОТОВОЙ МИССИИ.
        # Спутник уже вышел на орбиту Муны (6531 очко), а строка отчёта
        # уронила полёт на `AttributeError` и пометила его прерванным.
        # Записи ведутся через `fact`, как и всё остальное в этом файле.
        try:
            if ok and self.flight_log:
                self.flight_log.fact(f"Спутник поставлен в точку {phase:.0f}°")
        except Exception as exc:
            log.warning("Не удалось записать отчёт о постановке: %s", exc)
        return ok

    # ------------------------------------------------------------------
    def _penalise_wasted_fuel(self) -> None:
        if self.telemetry is None:
            return
        snap = self.telemetry.snapshot()
        lf = snap.resources.get("LiquidFuel", {})
        if lf.get("max", 0) <= 0:
            return
        left = lf.get("amount", 0.0) / lf["max"]
        if left < 0.15 and Milestone.MUN_ORBIT not in self.rewards.achieved:
            self.rewards.penalise(Failure.FUEL_WASTED, snap.mission_time,
                                  {"fuel_left_fraction": round(left, 3)})

    def _finish(self, phase: Phase) -> dict:
        self._enter(phase)
        extra = {"final_phase": phase.value, "phase_log": self.phase_log}
        try:
            if self.telemetry is not None:
                snap = self.telemetry.snapshot(detailed=True)
                self.final_snapshot = snap
                extra["telemetry_final"] = snap.to_dict()
                if self.flight_log:
                    self.flight_log.telemetry(snap)
        except Exception as exc:
            log.debug("Не удалось снять финальную телеметрию: %s", exc)

        try:
            if self.control is not None:
                self.control.full_stop()
            if self.telemetry is not None:
                self.telemetry.close()
        except Exception:
            pass

        extra["wall_time_s"] = round(time.time() - self.started_wall, 1)
        summary = self.rewards.finalize(extra)
        if self.flight_log:
            self.flight_log.status(
                "миссия выполнена" if phase is Phase.COMPLETE else "миссия прервана")
            self.flight_log.finish(summary)
        return summary


# Обратная совместимость с прежним именем
MunMission = Mission
