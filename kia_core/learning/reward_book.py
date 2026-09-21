"""Прейскурант наград — редактируемый оператором на ходу.

Оператор из консоли KIA может менять любую цену («повысь штраф за взрыв
до -2000») и добавлять собственные правила («дай +500 за ровный шаг
тангажа»). Изменения сразу действуют на текущее и все следующие поколения
и сохраняются в data/reward_book.json.
"""
from __future__ import annotations

import json
import math
import statistics
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import DATA_DIR
from ..logging_setup import get_logger

log = get_logger("learning.reward_book")

BOOK_FILE = DATA_DIR / "reward_book.json"

# --------------------------------------------------------------------------
# Базовый прейскурант
# --------------------------------------------------------------------------
DEFAULT_MILESTONES: dict[str, float] = {
    "viable_design": 50.0,
    "liftoff": 100.0,
    "cleared_tower": 25.0,
    "supersonic": 25.0,
    "atmosphere_exit": 150.0,
    "stable_orbit": 500.0,
    "mun_node_created": 1000.0,
    "mun_encounter": 500.0,
    "mun_soi_entered": 1500.0,
    "mun_orbit": 2500.0,
    "target_landing": 3000.0,
    "science_collected": 40.0,
    "crew_survived": 200.0,
    "returned_home": 1500.0,
}

DEFAULT_FAILURES: dict[str, float] = {
    "design_rejected": -200.0,
    "no_liftoff": -300.0,
    "explosion": -1000.0,
    "crash": -1000.0,
    "out_of_fuel": -400.0,
    "fuel_wasted": -200.0,
    "out_of_electricity": -250.0,
    "timeout": -200.0,
    "lost_control": -350.0,
    "connection_lost": -200.0,
    "unexpected_error": -200.0,
}

DEFAULT_TERMINAL = [
    "design_rejected", "no_liftoff", "explosion", "crash", "out_of_fuel",
    "out_of_electricity", "timeout", "lost_control", "connection_lost",
    "unexpected_error",
]

# Русские названия для команд оператора -> ключ прейскуранта
ALIASES: dict[str, str] = {
    # достижения
    "конструкция": "viable_design", "проект": "viable_design",
    "отрыв": "liftoff", "старт": "liftoff", "взлет": "liftoff",
    "вышка": "cleared_tower", "башня": "cleared_tower",
    "сверхзвук": "supersonic",
    "космос": "atmosphere_exit", "атмосфера": "atmosphere_exit",
    "орбита": "stable_orbit", "орбиту": "stable_orbit",
    "узел": "mun_node_created", "маневр": "mun_node_created",
    "перехват": "mun_encounter", "встреча": "mun_encounter",
    "сои": "mun_soi_entered", "сои муны": "mun_soi_entered",
    "муна": "mun_orbit", "орбита муны": "mun_orbit", "мун": "mun_orbit",
    "посадка": "target_landing",
    "наука": "science_collected",
    "экипаж": "crew_survived",
    "возвращение": "returned_home", "домой": "returned_home",
    # отказы
    "взрыв": "explosion",
    "падение": "crash", "краш": "crash", "разбился": "crash",
    "топливо": "out_of_fuel", "без топлива": "out_of_fuel",
    "перерасход": "fuel_wasted", "трата топлива": "fuel_wasted",
    "электричество": "out_of_electricity", "батарея": "out_of_electricity",
    "таймаут": "timeout", "время": "timeout",
    "управление": "lost_control", "потеря управления": "lost_control",
    "связь": "connection_lost",
    "ошибка": "unexpected_error",
    "отклонен": "design_rejected", "отклонена": "design_rejected",
    "не взлетел": "no_liftoff",
}


# --------------------------------------------------------------------------
# Формирующие правила: метрики качества полёта
# --------------------------------------------------------------------------
@dataclass
class CustomRule:
    """Правило вида «дать N очков за качество X»."""
    key: str
    weight: float
    description: str
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# Каталог доступных метрик: ключ -> (описание, русские синонимы)
RULE_CATALOG: dict[str, tuple[str, tuple[str, ...]]] = {
    "smooth_pitch": ("ровный шаг тангажа на выведении",
                     ("тангаж", "ровный тангаж", "плавный тангаж", "шаг тангажа")),
    "fuel_efficiency": ("остаток топлива в конце миссии",
                        ("экономия", "остаток топлива", "экономия топлива")),
    "low_gforce": ("мягкий полёт без больших перегрузок",
                   ("перегрузка", "перегрузки", "мягкий полет")),
    "fast_orbit": ("быстрый выход на орбиту",
                   ("быстрая орбита", "скорость выведения")),
    "accurate_orbit": ("круговая, точная орбита",
                       ("точная орбита", "круговая орбита", "круговость")),
    "gentle_maxq": ("аккуратное прохождение максимального напора",
                    ("напор", "максq", "макс напор")),
    "light_vehicle": ("лёгкая ракета (экономия массы)",
                      ("масса", "легкая ракета", "лёгкая ракета")),
    "few_parts": ("простая конструкция (мало деталей)",
                  ("детали", "простота", "мало деталей")),
}


class FlightMetrics:
    """Копит телеметрию за полёт и считает по ней качество (0..1)."""

    def __init__(self):
        self.pitch_samples: list[tuple[float, float]] = []   # (время, тангаж)
        self.max_g: float = 0.0
        self.max_q: float = 0.0
        self.orbit_time: float | None = None
        self.fuel_left_fraction: float = 1.0
        self.circularity: float = 0.0
        self.total_mass: float = 0.0
        self.part_count: int = 0

    # ---------------- сбор ----------------
    def observe(self, snap) -> None:
        if snap.mission_time > 0 and snap.altitude < 80_000:
            self.pitch_samples.append((snap.mission_time, snap.pitch))
        self.max_g = max(self.max_g, snap.g_force)
        self.max_q = max(self.max_q, snap.dynamic_pressure)
        if self.total_mass == 0.0:
            self.total_mass = snap.mass
        self.part_count = max(self.part_count, snap.part_count)
        lf = snap.resources.get("LiquidFuel", {})
        if lf.get("max", 0) > 0:
            self.fuel_left_fraction = lf["amount"] / lf["max"]
        if snap.apoapsis > 0 and snap.periapsis > 0:
            self.circularity = (min(snap.apoapsis, snap.periapsis)
                                / max(snap.apoapsis, snap.periapsis))

    def mark_orbit(self, mission_time: float) -> None:
        if self.orbit_time is None:
            self.orbit_time = mission_time

    # ---------------- метрики (0..1) ----------------
    def smooth_pitch(self) -> float:
        """1.0 — тангаж меняется равномерно, 0 — рывками."""
        if len(self.pitch_samples) < 8:
            return 0.0
        rates = []
        for (t0, p0), (t1, p1) in zip(self.pitch_samples, self.pitch_samples[1:]):
            dt = t1 - t0
            if dt > 1e-3:
                rates.append((p1 - p0) / dt)
        if len(rates) < 4:
            return 0.0
        spread = statistics.pstdev(rates)
        mean = abs(statistics.fmean(rates)) + 1e-6
        # коэффициент вариации: чем меньше, тем ровнее шаг
        cv = spread / mean
        return max(0.0, min(1.0, math.exp(-cv / 2.0)))

    def fuel_efficiency(self) -> float:
        return max(0.0, min(1.0, self.fuel_left_fraction))

    def low_gforce(self) -> float:
        if self.max_g <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.max_g - 1.0) / 5.0))

    def fast_orbit(self) -> float:
        if self.orbit_time is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.orbit_time - 240.0) / 600.0))

    def accurate_orbit(self) -> float:
        return max(0.0, min(1.0, (self.circularity - 0.8) / 0.2)) if self.circularity else 0.0

    def gentle_maxq(self) -> float:
        if self.max_q <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.max_q - 15_000.0) / 30_000.0))

    def light_vehicle(self) -> float:
        if self.total_mass <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.total_mass - 10.0) / 60.0))

    def few_parts(self) -> float:
        if self.part_count <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.part_count - 15) / 60.0))

    def value(self, key: str) -> float:
        fn = getattr(self, key, None)
        return float(fn()) if callable(fn) else 0.0

    def snapshot(self) -> dict:
        return {key: round(self.value(key), 3) for key in RULE_CATALOG}


# --------------------------------------------------------------------------
class RewardBook:
    """Живой прейскурант: цены достижений, штрафов и пользовательских правил."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or BOOK_FILE)
        self._lock = threading.RLock()
        self.milestones: dict[str, float] = dict(DEFAULT_MILESTONES)
        self.failures: dict[str, float] = dict(DEFAULT_FAILURES)
        self.terminal: set[str] = set(DEFAULT_TERMINAL)
        self.rules: dict[str, CustomRule] = {}
        self.load()

    # ---------------- доступ ----------------
    def milestone(self, key: str) -> float:
        return self.milestones.get(key, 0.0)

    def failure(self, key: str) -> float:
        return self.failures.get(key, 0.0)

    def is_terminal(self, key: str) -> bool:
        return key in self.terminal

    def max_possible(self) -> float:
        return (sum(v for v in self.milestones.values() if v > 0)
                + sum(r.weight for r in self.rules.values() if r.enabled and r.weight > 0))

    # ---------------- изменение ----------------
    def resolve(self, name: str) -> str | None:
        """Ключ прейскуранта по русскому названию или самому ключу."""
        key = name.strip().lower().replace("ё", "е")
        if key in self.milestones or key in self.failures or key in RULE_CATALOG:
            return key
        if key in ALIASES:
            return ALIASES[key]
        if not key:
            return None
        # синоним правила целиком либо как часть фразы («ровный шаг тангажа»)
        for rule_key, (_, synonyms) in RULE_CATALOG.items():
            for synonym in synonyms:
                if key == synonym or synonym in key or key in synonym:
                    return rule_key
        # русский псевдоним внутри фразы («штраф за взрыв на старте»)
        for alias, candidate in ALIASES.items():
            if key == alias or alias in key.split() or f" {alias} " in f" {key} ":
                return candidate
        # частичное совпадение по ключам прейскуранта
        for candidate in list(self.milestones) + list(self.failures) + list(RULE_CATALOG):
            if key in candidate or candidate.startswith(key):
                return candidate
        for alias, candidate in ALIASES.items():
            if key in alias:
                return candidate
        # сравнение по основам слов: «экономию топлива» ≈ «экономия топлива»
        stemmed = _stems(key)
        if stemmed:
            for rule_key, (_, synonyms) in RULE_CATALOG.items():
                for synonym in synonyms:
                    if stemmed & _stems(synonym):
                        return rule_key
            for alias, candidate in ALIASES.items():
                if stemmed & _stems(alias):
                    return candidate
        return None

    def set_value(self, name: str, value: float) -> tuple[bool, str]:
        """Меняет цену. Возвращает (успех, сообщение для оператора)."""
        key = self.resolve(name)
        if key is None:
            return False, (f"Не понял, за что менять цену: «{name}». "
                           f"Список: команда `награды`.")
        with self._lock:
            if key in RULE_CATALOG:
                description = RULE_CATALOG[key][0]
                self.rules[key] = CustomRule(key, float(value), description)
                self.save()
                return True, f"Правило «{description}» теперь стоит {value:+.0f} очков"
            if key in self.milestones or value >= 0:
                old = self.milestones.get(key, 0.0)
                self.milestones[key] = float(value)
                self.save()
                return True, f"Награда «{key}»: {old:+.0f} → {value:+.0f} очков"
            old = self.failures.get(key, 0.0)
            self.failures[key] = float(value)
            self.save()
            return True, f"Штраф «{key}»: {old:+.0f} → {value:+.0f} очков"

    def set_terminal(self, name: str, terminal: bool) -> tuple[bool, str]:
        key = self.resolve(name)
        if key is None:
            return False, f"Неизвестное событие: «{name}»"
        with self._lock:
            if terminal:
                self.terminal.add(key)
            else:
                self.terminal.discard(key)
            self.save()
        return True, (f"Событие «{key}» "
                      f"{'завершает' if terminal else 'больше не завершает'} сессию")

    def remove_rule(self, name: str) -> tuple[bool, str]:
        key = self.resolve(name)
        with self._lock:
            if key and key in self.rules:
                del self.rules[key]
                self.save()
                return True, f"Правило «{key}» убрано"
        return False, f"Правила «{name}» нет в списке"

    def reset(self) -> None:
        with self._lock:
            self.milestones = dict(DEFAULT_MILESTONES)
            self.failures = dict(DEFAULT_FAILURES)
            self.terminal = set(DEFAULT_TERMINAL)
            self.rules = {}
            self.save()

    # ---------------- расчёт правил ----------------
    def evaluate_rules(self, metrics: FlightMetrics) -> list[tuple[str, float, float]]:
        """Возвращает [(ключ, качество 0..1, очки)] по накопленным метрикам."""
        out = []
        with self._lock:
            rules = [r for r in self.rules.values() if r.enabled]
        for rule in rules:
            quality = metrics.value(rule.key)
            out.append((rule.key, quality, rule.weight * quality))
        return out

    # ---------------- сохранение ----------------
    def to_dict(self) -> dict:
        return {
            "milestones": self.milestones,
            "failures": self.failures,
            "terminal": sorted(self.terminal),
            "rules": {k: r.to_dict() for k, r in self.rules.items()},
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Прейскурант не прочитан (%s) — беру значения по умолчанию", exc)
            return
        self.milestones.update({k: float(v) for k, v in data.get("milestones", {}).items()})
        self.failures.update({k: float(v) for k, v in data.get("failures", {}).items()})
        if "terminal" in data:
            self.terminal = set(data["terminal"])
        for key, raw in data.get("rules", {}).items():
            self.rules[key] = CustomRule(
                key=raw.get("key", key), weight=float(raw.get("weight", 0.0)),
                description=raw.get("description", RULE_CATALOG.get(key, ("",))[0]),
                enabled=bool(raw.get("enabled", True)))
        log.info("Прейскурант загружен: %d наград, %d штрафов, %d правил",
                 len(self.milestones), len(self.failures), len(self.rules))

    # ---------------- отображение ----------------
    def describe(self) -> str:
        lines = ["ДОСТИЖЕНИЯ:"]
        for key, value in sorted(self.milestones.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {value:+8.0f}  {key}")
        lines.append("ШТРАФЫ:")
        for key, value in sorted(self.failures.items(), key=lambda kv: kv[1]):
            mark = " (обрывает сессию)" if key in self.terminal else ""
            lines.append(f"  {value:+8.0f}  {key}{mark}")
        if self.rules:
            lines.append("ПОЛЬЗОВАТЕЛЬСКИЕ ПРАВИЛА:")
            for rule in self.rules.values():
                state = "" if rule.enabled else " (выключено)"
                lines.append(f"  {rule.weight:+8.0f}  {rule.key} — {rule.description}{state}")
        available = [k for k in RULE_CATALOG if k not in self.rules]
        if available:
            lines.append("ДОСТУПНЫЕ ПРАВИЛА (можно назначить цену):")
            for key in available:
                lines.append(f"           {key} — {RULE_CATALOG[key][0]}")
        lines.append(f"Максимум за идеальную миссию: {self.max_possible():.0f} очков")
        return "\n".join(lines)


def _stems(phrase: str) -> set[str]:
    """Основы слов длиной 5 символов — грубая замена морфологии."""
    return {word[:5] for word in phrase.lower().replace("ё", "е").split()
            if len(word) >= 5}


# Единый прейскурант процесса
REWARD_BOOK = RewardBook()
