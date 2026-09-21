"""Геном особи: инженерная политика + профиль полёта.

Раньше геном перебирал индексы деталей. Теперь детали подбирает сам
конструктор под задачу, а эволюция настраивает ЕГО ПОЛИТИКУ (сколько ΔV
отдать первой ступени, какой TWR считать целевым, какой держать запас)
и профиль полёта пилота.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..engineer.autodesign import DesignPolicy


@dataclass
class FlightGenome:
    """Профиль выведения и манёвров (непрерывные параметры)."""
    turn_start_altitude: float = 250.0      # м, начало гравитационного разворота
    turn_end_altitude: float = 45_000.0     # м, конец разворота
    turn_exponent: float = 0.55             # форма кривой тангажа
    target_apoapsis: float = 80_000.0       # м
    ascent_heading: float = 90.0            # азимут (90 = на восток)
    max_q_throttle: float = 0.75            # ограничение тяги в зоне напора
    max_q_threshold: float = 18_000.0       # Па, порог ограничения
    coast_throttle: float = 0.05            # поддержка апоапсиса на выбеге
    circularization_lead: float = 0.5       # доля ожога до апоапсиса
    node_execute_lead: float = 0.5
    transfer_phase_bias: float = 0.0        # градусы поправки к фазовому углу
    mun_periapsis_target: float = 30_000.0  # м
    capture_lead: float = 0.5

    @staticmethod
    def bounds() -> dict[str, tuple[float, float]]:
        return {
            "turn_start_altitude": (100.0, 3_000.0),
            "turn_end_altitude": (25_000.0, 70_000.0),
            "turn_exponent": (0.25, 1.30),
            "ascent_heading": (80.0, 100.0),
            "max_q_throttle": (0.35, 1.00),
            "max_q_threshold": (8_000.0, 40_000.0),
            "coast_throttle": (0.0, 0.30),
            "circularization_lead": (0.35, 0.65),
            "node_execute_lead": (0.35, 0.65),
            "transfer_phase_bias": (-15.0, 15.0),
            "capture_lead": (0.35, 0.65),
        }

    def to_dict(self) -> dict:
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "FlightGenome":
        known = {k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def clamp(self) -> "FlightGenome":
        for name, (lo, hi) in self.bounds().items():
            setattr(self, name, max(lo, min(hi, float(getattr(self, name)))))
        return self


@dataclass
class Individual:
    """Особь популяции: политика конструктора + план полёта + результат."""
    design: DesignPolicy = field(default_factory=DesignPolicy)
    flight: FlightGenome = field(default_factory=FlightGenome)
    score: float | None = None
    generation: int = 0
    index: int = 0
    parents: tuple = ()

    def to_dict(self) -> dict:
        return {
            "design": self.design.to_dict(),
            "flight": self.flight.to_dict(),
            "score": self.score,
            "generation": self.generation,
            "index": self.index,
            "parents": list(self.parents),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        return cls(
            design=DesignPolicy.from_dict(data.get("design", {})),
            flight=FlightGenome.from_dict(data.get("flight", {})),
            score=data.get("score"),
            generation=int(data.get("generation", 0)),
            index=int(data.get("index", 0)),
            parents=tuple(data.get("parents", [])),
        )

    def copy(self) -> "Individual":
        return Individual.from_dict(self.to_dict())

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Individual":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data.get("genome", data))

    def describe(self) -> str:
        d, f = self.design, self.flight
        return (f"G{self.generation}#{self.index} | "
                f"ΔV 1-й ступени {d.ascent_split:.0%}, TWR старта {d.liftoff_twr:.2f}, "
                f"запас ×{d.dv_margin:.2f} | разворот "
                f"{f.turn_start_altitude:.0f}→{f.turn_end_altitude:.0f} м "
                f"(n={f.turn_exponent:.2f}), Ap={f.target_apoapsis/1000:.0f} км")


# --------------------------------------------------------------------------
def _sections(ind: Individual):
    return ((ind.design, DesignPolicy.bounds()), (ind.flight, FlightGenome.bounds()))


def random_individual(rng: random.Random, generation: int = 0, index: int = 0) -> Individual:
    ind = Individual(generation=generation, index=index)
    for section, bounds in _sections(ind):
        for name, (lo, hi) in bounds.items():
            setattr(section, name, rng.uniform(lo, hi))
    ind.design.clamp()
    ind.flight.clamp()
    return ind


def jitter_individual(base: Individual, rng: random.Random, scale: float = 0.1,
                      generation: int = 0, index: int = 0) -> Individual:
    """Небольшое возмущение вокруг известной хорошей особи."""
    child = base.copy()
    child.score = None
    child.generation, child.index = generation, index
    child.parents = (base.generation, base.index)
    for section, bounds in _sections(child):
        for name, (lo, hi) in bounds.items():
            value = getattr(section, name) + rng.gauss(0.0, (hi - lo) * scale)
            setattr(section, name, max(lo, min(hi, value)))
    return child


# ==========================================================================
# Закрепление конструкции оператором
# ==========================================================================
DESIGN_LOCK_NAME = "design_lock.json"


def design_lock_path() -> "Path":
    from pathlib import Path
    from ..config import CONFIG
    return Path(getattr(CONFIG, "data_dir", "data")) / DESIGN_LOCK_NAME


def save_design_lock(policy: DesignPolicy) -> "Path":
    """Запоминает конструкцию, которую оператор одобрил в ангаре.

    Пока файл существует, автономный цикл строит РОВНО такую ракету, а
    эволюция ищет только профиль полёта. Без этого оператор видит в
    ангаре одну машину, а в полёте другую: `craft` собирает по настройкам
    по умолчанию, а цикл — по геному эволюции.
    """
    import json
    path = design_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def load_design_lock() -> DesignPolicy | None:
    """Закреплённая конструкция или None, если оператор её не закреплял."""
    import json
    path = design_lock_path()
    if not path.exists():
        return None
    try:
        return DesignPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def clear_design_lock() -> bool:
    path = design_lock_path()
    if path.exists():
        path.unlink()
        return True
    return False
