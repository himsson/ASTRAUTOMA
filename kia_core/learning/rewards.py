"""Система наград и штрафов KIA.

Цены не зашиты в код: они берутся из живого прейскуранта (RewardBook),
который оператор правит прямо во время работы системы. Здесь остаётся
логика — когда что начислять, что обрывает сессию и как считать
формирующие (непрерывные) награды.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..logging_setup import get_logger
from .reward_book import REWARD_BOOK, FlightMetrics, RewardBook

log = get_logger("learning.rewards")


class Milestone(str, Enum):
    VIABLE_DESIGN = "viable_design"
    LIFTOFF = "liftoff"
    CLEARED_TOWER = "cleared_tower"
    SUPERSONIC = "supersonic"
    ATMOSPHERE_EXIT = "atmosphere_exit"
    STABLE_ORBIT = "stable_orbit"
    MUN_NODE_CREATED = "mun_node_created"
    MUN_ENCOUNTER = "mun_encounter"
    MUN_SOI_ENTERED = "mun_soi_entered"
    MUN_ORBIT = "mun_orbit"
    TARGET_LANDING = "target_landing"
    SCIENCE_COLLECTED = "science_collected"
    CREW_SURVIVED = "crew_survived"
    RETURNED_HOME = "returned_home"


class Failure(str, Enum):
    DESIGN_REJECTED = "design_rejected"
    NO_LIFTOFF = "no_liftoff"
    EXPLOSION = "explosion"
    CRASH = "crash"
    OUT_OF_FUEL = "out_of_fuel"
    FUEL_WASTED = "fuel_wasted"
    OUT_OF_ELECTRICITY = "out_of_electricity"
    TIMEOUT = "timeout"
    LOST_CONTROL = "lost_control"
    CONNECTION_LOST = "connection_lost"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass
class ScoreEvent:
    label: str
    points: float
    kind: str                 # milestone | failure | shaping | rule
    mission_time: float = 0.0
    wall_time: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RewardSystem:
    """Начисляет очки за прогресс миссии и штрафы за отказы."""

    def __init__(self, book: RewardBook | None = None, flight_log=None,
                 spec=None):
        from ..mission_spec import MissionSpec
        self.book = book or REWARD_BOOK
        self.flight_log = flight_log
        self.spec = spec or MissionSpec()
        self.metrics = FlightMetrics()
        self.events: list[ScoreEvent] = []
        self.total: float = 0.0
        self.achieved: set[Milestone] = set()
        self.failures: list[Failure] = []
        self.terminated: bool = False
        self.termination_reason: str | None = None
        self._best_altitude = 0.0
        self._best_apoapsis = 0.0
        # Лучший периапсис за полёт. Начинаем «с самого дна»: любой
        # суборбитальный аппарат стартует с глубоко отрицательным.
        self._best_periapsis = -1e9

    # ------------------------------------------------------------------
    def award(self, milestone: Milestone, mission_time: float = 0.0,
              detail: dict | None = None) -> float:
        if milestone in self.achieved:
            return 0.0
        points = self.book.milestone(milestone.value)
        self.achieved.add(milestone)
        self.total += points
        self.events.append(ScoreEvent(milestone.value, points, "milestone",
                                      mission_time, detail=detail or {}))
        log.info("[%+.0f] %s (итого %.0f)", points, milestone.value, self.total)
        if self.flight_log:
            self.flight_log.reward(f"Достигнуто: {milestone.value}", points,
                                   _fmt_detail(detail))
        return points

    def penalise(self, failure: Failure, mission_time: float = 0.0,
                 detail: dict | None = None, scale: float = 1.0) -> float:
        points = self.book.failure(failure.value) * scale
        self.total += points
        self.failures.append(failure)
        self.events.append(ScoreEvent(failure.value, points, "failure",
                                      mission_time, detail=detail or {}))
        log.warning("[%+.0f] %s (итого %.0f)", points, failure.value, self.total)
        if self.flight_log:
            self.flight_log.reward(f"Отказ: {failure.value}", points, _fmt_detail(detail))
        if self.book.is_terminal(failure.value):
            self.terminated = True
            self.termination_reason = failure.value
        return points

    def shape(self, label: str, points: float, mission_time: float = 0.0,
              detail: dict | None = None) -> float:
        self.total += points
        self.events.append(ScoreEvent(label, points, "shaping", mission_time,
                                      detail=detail or {}))
        return points

    # ------------------------------------------------------------------
    # Непрерывные компоненты
    # ------------------------------------------------------------------
    def shape_altitude(self, altitude: float, mission_time: float = 0.0) -> float:
        if altitude <= self._best_altitude:
            return 0.0
        ceiling = max(70_000.0, self.spec.park_orbit_altitude)
        gained = min(altitude, ceiling) - min(self._best_altitude, ceiling)
        self._best_altitude = altitude
        if gained <= 0:
            return 0.0
        return self.shape("altitude_progress", gained / 1000.0, mission_time)

    def shape_apoapsis(self, apoapsis: float, target: float,
                       mission_time: float = 0.0) -> float:
        if apoapsis <= self._best_apoapsis or target <= 0:
            return 0.0
        prev = min(self._best_apoapsis, target)
        now = min(apoapsis, target)
        self._best_apoapsis = apoapsis
        gain = (now - prev) / target * 100.0
        return self.shape("apoapsis_progress", gain, mission_time) if gain > 0 else 0.0

    def shape_orbit_quality(self, apoapsis: float, periapsis: float,
                            mission_time: float = 0.0) -> float:
        if periapsis <= 0 or apoapsis <= 0:
            return 0.0
        ratio = min(apoapsis, periapsis) / max(apoapsis, periapsis)
        return self.shape("orbit_circularity", 100.0 * ratio ** 2, mission_time)

    # Насколько глубоко периапсис уходит в минус, прежде чем считать
    # аппарат «даже не начавшим». Радиус Кербина: ниже этого он падает
    # почти вертикально.
    DEEP_SUBORBITAL = 600_000.0

    def shape_periapsis(self, apoapsis: float, periapsis: float,
                        target_apoapsis: float, mission_time: float = 0.0) -> float:
        """Насколько близко аппарат подошёл к ЗАМКНУТОЙ орбите.

        Дыра, которую это закрывает. `shape_orbit_quality` возвращает ноль
        при любом отрицательном периапсисе, а он отрицателен во всех
        полётах до самой орбиты. То есть весь суборбитальный диапазон для
        оптимизатора плоский: аппарат с периапсисом -138 км, которому до
        орбиты остался один манёвр, и аппарат с -585 км, который просто
        подпрыгнул, получают одинаковый ноль.

        Замер: полёт со скруглением, отработавшим 1299 м/с из 1528 и
        периапсисом -138 км, набрал 566 очков. Соседний с периапсисом
        -585 км набрал 564. Разницу в качестве счёт не увидел, и поиск
        заметался.

        Награда даётся только когда апоапсис уже около цели: поднимать
        периапсис, не выйдя на высоту, смысла нет, и платить за это
        нельзя — иначе система найдёт вырожденное решение «лететь низко
        и ровно».
        """
        if apoapsis < 0.7 * target_apoapsis or periapsis >= 0:
            return 0.0
        # Ратчет, как у высоты и апоапсиса: платим только за УЛУЧШЕНИЕ,
        # иначе награда капала бы каждый тик и завалила бы остальной счёт.
        if periapsis <= self._best_periapsis:
            return 0.0
        prev = max(self._best_periapsis, -self.DEEP_SUBORBITAL)
        now = max(periapsis, -self.DEEP_SUBORBITAL)
        self._best_periapsis = periapsis
        gain = (now - prev) / self.DEEP_SUBORBITAL * 100.0
        if gain <= 0:
            return 0.0
        return self.shape("periapsis_rise", gain, mission_time,
                          {"periapsis": round(periapsis),
                           "apoapsis": round(apoapsis)})

    def shape_target_proximity(self, distance_m: float, mission_time: float = 0.0) -> float:
        if distance_m <= 0 or distance_m == float("inf"):
            return 0.0
        reward = 300.0 * min(1.0, 2.4e6 / max(distance_m, 2.0e5))
        return self.shape("target_proximity", reward, mission_time)

    # ------------------------------------------------------------------
    # Оценка телеметрии
    # ------------------------------------------------------------------
    def observe(self, snap) -> None:
        """Копит метрики качества полёта (для пользовательских правил)."""
        self.metrics.observe(snap)

    def evaluate_telemetry(self, snap, target_apoapsis: float | None = None) -> None:
        self.observe(snap)
        target_apoapsis = target_apoapsis or self.spec.park_orbit_altitude
        mt = snap.mission_time
        home = self.spec.home_body
        target = self.spec.target_body

        if snap.altitude > 500 and snap.situation in ("flying", "sub_orbital"):
            self.award(Milestone.CLEARED_TOWER, mt, {"altitude": round(snap.altitude)})
        if snap.speed > 340 and snap.altitude > 5_000:
            self.award(Milestone.SUPERSONIC, mt, {"speed": round(snap.speed)})
        if snap.altitude > 70_000 and snap.body == home:
            self.award(Milestone.ATMOSPHERE_EXIT, mt, {"altitude": round(snap.altitude)})

        self.shape_altitude(snap.altitude, mt)
        self.shape_apoapsis(snap.apoapsis, target_apoapsis, mt)
        self.shape_periapsis(snap.apoapsis, snap.periapsis, target_apoapsis, mt)

        # Опорная орбита у домашнего тела
        if snap.body == home and snap.periapsis > 70_000 and snap.apoapsis > 70_000:
            ratio = min(snap.apoapsis, snap.periapsis) / max(snap.apoapsis, snap.periapsis)
            if ratio > 0.85:
                if Milestone.STABLE_ORBIT not in self.achieved:
                    self.metrics.mark_orbit(mt)
                self.award(Milestone.STABLE_ORBIT, mt,
                           {"apoapsis": round(snap.apoapsis),
                            "periapsis": round(snap.periapsis),
                            "circularity": round(ratio, 3)})

        # Цель
        if target != home and snap.body == target:
            self.award(Milestone.MUN_SOI_ENTERED, mt, {"body": target})
            if snap.periapsis > 0 and 0 < snap.eccentricity < 1.0:
                self.award(Milestone.MUN_ORBIT, mt,
                           {"apoapsis": round(snap.apoapsis),
                            "periapsis": round(snap.periapsis),
                            "eccentricity": round(snap.eccentricity, 3)})
            if snap.situation in ("landed", "splashed"):
                if self.spec.needs_landing:
                    self.award(Milestone.TARGET_LANDING, mt, {"body": target})
                else:
                    self.penalise(Failure.CRASH, mt, {"body": target})

        # Возвращение домой
        if (self.spec.needs_return and snap.body == home
                and snap.situation in ("landed", "splashed")
                and Milestone.STABLE_ORBIT in self.achieved and mt > 300):
            self.award(Milestone.RETURNED_HOME, mt, {"body": home})

        ec = snap.resources.get("ElectricCharge", {})
        if ec.get("max", 0) > 0 and ec.get("amount", 1) < 5.0:
            self.penalise(Failure.OUT_OF_ELECTRICITY, mt,
                          {"charge": round(ec.get("amount", 0), 2)})

    # ------------------------------------------------------------------
    def apply_custom_rules(self) -> list[dict]:
        """Начисляет очки по пользовательским правилам прейскуранта."""
        applied = []
        for key, quality, points in self.book.evaluate_rules(self.metrics):
            self.total += points
            self.events.append(ScoreEvent(key, points, "rule",
                                          detail={"quality": round(quality, 3)}))
            applied.append({"rule": key, "quality": round(quality, 3),
                            "points": round(points, 1)})
            log.info("[%+.0f] правило «%s» (качество %.2f)", points, key, quality)
            if self.flight_log:
                self.flight_log.reward(f"Правило «{key}»", points,
                                       f"качество {quality:.2f}")
        return applied

    # ------------------------------------------------------------------
    def finalize(self, extra: dict | None = None) -> dict:
        rules = self.apply_custom_rules()
        result = {
            "total_score": round(self.total, 2),
            "milestones": sorted(m.value for m in self.achieved),
            "failures": [f.value for f in self.failures],
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "events": [e.to_dict() for e in self.events],
            "custom_rules": rules,
            "metrics": self.metrics.snapshot(),
            "peak_altitude": self._best_altitude,
            "peak_apoapsis": self._best_apoapsis,
            "mission": self.spec.describe(),
        }
        if extra:
            result.update(extra)
        log.info("Сессия завершена. Счёт: %.1f | достижений: %d | отказов: %d",
                 self.total, len(self.achieved), len(self.failures))
        return result

    # ------------------------------------------------------------------
    def max_possible_score(self) -> float:
        return self.book.max_possible()

    def progress_fraction(self) -> float:
        earned = sum(self.book.milestone(m.value) for m in self.achieved)
        top = self.book.max_possible()
        return earned / top if top else 0.0


def _fmt_detail(detail: dict | None) -> str:
    if not detail:
        return ""
    return ", ".join(f"{k}={v}" for k, v in detail.items())
