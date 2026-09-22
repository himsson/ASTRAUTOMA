"""What the designer learned from other people's craft (Builder weights).

The Builder module carries `design_school.json`: statistics over real
craft from the Steam Workshop, the stock game and the player's own saves
— only craft whose every part exists in this game, no planes. From it:

    rockets     stages for a Δv class, liftoff and upper-stage TWR, how the
                Δv is split between stages, engines people put on each
                stage, how often boosters, fairings and fins are used
    rover / base / station / probe
                what such a craft carries: wheels, batteries, panels,
                antennas, docking ports, lights, crew seats, science

Nothing here is invented: when the school has no data for a question, the
caller keeps its own default.
"""
from __future__ import annotations

import json
import math

from .i18n import L

KINDS = ("rocket", "rover", "base", "station", "probe")
_CACHE: dict = {}


def data() -> dict:
    from .knowledge import INSTALLED
    path = INSTALLED / "Builder" / "design_school.json"
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return {}
    if _CACHE.get("stamp") != stamp:
        try:
            _CACHE.update(stamp=stamp, data=json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {}
    return _CACHE["data"]


def learned() -> bool:
    return bool(data().get("rocket", {}).get("classes"))


def kind_title(kind: str) -> str:
    return {"rocket": L("rocket only (payload is yours)", "только ракета (нагрузка ваша)"),
            "rover": L("rover", "ровер"), "base": L("surface base", "база на поверхности"),
            "station": L("space station", "космическая станция"),
            "probe": L("telescope / probe", "телескоп / зонд")}[kind]


# --- rockets -------------------------------------------------------------------
def rocket_class(dv: float) -> dict:
    classes = data().get("rocket", {}).get("classes", {})
    for c in classes.values():
        lo, hi = c.get("dv_range", (0, 1e9))
        if lo <= dv < hi:
            return c
    return {}


def stages_for(dv: float) -> int | None:
    c = rocket_class(dv)
    per = data().get("rocket", {}).get("dv_per_stage")
    if not c:
        return None
    n = int(c.get("typical_stages", 2))
    if per:
        n = round((n + dv / per) / 2)            # what people build, nudged by the Δv itself
    return max(2, min(5, n))


def _median(band, default):
    return float(band["median"]) if band and band.get("median") is not None else default


def twr_targets(dv: float, liftoff: float, upper: float) -> tuple[float, float, float]:
    """(liftoff, second stage, upper stages). The lower quartile of what people fly:
    enough to fly well, no heavier than needed."""
    c = rocket_class(dv)
    if not c:
        return liftoff, upper, upper
    lo = lambda b, d: float(b["p25"]) if b and b.get("p25") is not None else d   # noqa: E731
    t0 = min(2.0, max(1.3, lo(c.get("twr0"), liftoff)))
    t2 = min(1.4, max(0.6, lo(c.get("twr_second"), upper)))
    tu = min(1.2, max(0.5, lo(c.get("twr_upper"), upper)))
    return t0, t2, tu


def split_for(n: int, dv: float) -> list[float] | None:
    c = rocket_class(dv)
    sp = c.get("dv_split") if c else None
    if not sp or len(sp) != n or any(b is None for b in sp):
        return None
    shares = [max(0.05, _median(b, 1 / n)) for b in sp]
    total = sum(shares)
    return [s / total for s in shares]


def engine_bonus(role_index: int, name: str) -> float:
    """Score multiplier (< 1 is better) for engines people like on this stage."""
    roles = data().get("rocket", {}).get("engines_by_role", {})
    role = "first" if role_index == 0 else ("second" if role_index == 1 else "upper")
    counts = roles.get(role, {})
    if not counts or name not in counts:
        return 1.0
    top = max(counts.values())
    return 1.0 - 0.08 * math.sqrt(counts[name] / top)


def share(dv: float, key: str) -> float:
    c = rocket_class(dv)
    return float(c.get(key, 0.0)) if c else 0.0


def examples(kind: str, limit: int = 4) -> list[str]:
    rows = [c for c in data().get("crafts", []) if c.get("kind") == kind]
    rows.sort(key=lambda c: c.get("parts", 0))
    return [c["name"] for c in rows[:limit]]


def similar_rockets(dv: float, limit: int = 3) -> list[str]:
    rows = [c for c in data().get("crafts", []) if c.get("kind") == "rocket" and c.get("dv")]
    rows.sort(key=lambda c: abs(c["dv"] - dv))
    out = []
    for c in rows[:limit]:
        dv = f"{c['dv']:,}".replace(",", " ")
        out.append(L(f"{c['name']} ({dv} m/s, {c['stages']} stages)", f"{c['name']} ({dv} м/с, ступеней: {c['stages']})"))
    return out


# --- rovers, bases, stations, probes ------------------------------------------------
PARTS = {
    "wheel": ("roverWheel2", "roverWheel1"),
    "battery": ("batteryBankMini", "batteryPack"),
    "solar": ("solarPanels5", "solarPanels4"),
    "generator": ("rtg", "FuelCell"),
    "antenna": ("HighGainAntenna5_v2",),
    "docking": ("dockingPort2",),
    "light": ("groundLight2", "domeLight1"),
    "reaction_wheel": ("asasmodule1-2", "sasModule"),
    "science": ("sensorThermometer", "sensorBarometer", "GooExperiment", "science_module"),
    "habitat": ("crewCabin",),
    "isru": ("MiniDrill",),
    "leg": ("landingLeg1",),
    "rcs": ("RCSBlock_v2",),
}
TITLES = {
    "wheel": ("Wheels", "Колёса"), "battery": ("Batteries", "Аккумуляторы"),
    "solar": ("Solar panels", "Солнечные панели"), "generator": ("Generator", "Генератор"),
    "antenna": ("Antenna", "Антенна"), "docking": ("Docking ports", "Стыковочные узлы"),
    "light": ("Lights", "Фары"), "reaction_wheel": ("Reaction wheels", "Маховики"),
    "science": ("Science", "Наука"), "habitat": ("Crew modules", "Жилые модули"),
    "isru": ("Drill", "Бур"), "leg": ("Landing legs", "Опоры"), "rcs": ("RCS", "RCS"),
}
# A craft that fits one launch: people's giant workshop builds stretch the medians,
# so every count is capped at what one well-built craft of this kind carries.
CAP = {"wheel": 8, "battery": 4, "solar": 8, "generator": 2, "antenna": 2, "docking": 4, "light": 4,
       "reaction_wheel": 2, "science": 4, "habitat": 3, "isru": 1, "leg": 6, "rcs": 8}
MINIMUM = {"rover": {"wheel": 4, "battery": 1, "solar": 2, "light": 2, "reaction_wheel": 1},
           "base": {"leg": 4, "battery": 2, "solar": 4, "light": 2, "habitat": 1, "docking": 1},
           "station": {"docking": 2, "battery": 2, "solar": 4, "reaction_wheel": 1, "habitat": 1, "rcs": 4},
           "probe": {"battery": 1, "solar": 2, "antenna": 1, "science": 2, "reaction_wheel": 1}}


def composition(kind: str, part_lookup) -> list[tuple[str, str, int, str]]:
    """[(title, part name, count, why)] — the typical kit of this kind of craft."""
    comp = data().get(kind, {}).get("per_craft", {})
    base = MINIMUM.get(kind, {})
    out = []
    n_seen = data().get(kind, {}).get("n", 0)
    for key, names in PARTS.items():
        learned_n = int(round(float(comp.get(key, {}).get("median", 0)))) if comp else 0
        n = max(learned_n, base.get(key, 0))
        if key == "wheel" and n:
            n = max(4, n + (n % 2))
        if key in ("solar", "leg", "rcs", "light") and n % 2:
            n += 1
        n = min(n, CAP.get(key, 4))
        if n <= 0:
            continue
        why = (L(f"people put {learned_n} on such a craft ({n_seen} studied)",
                 f"так ставят люди: {learned_n} на аппарат ({n_seen} изучено)") if learned_n
               else L("minimum for this kind of craft", "минимум для такого аппарата"))
        en, ru = TITLES[key]
        if key == "science":                    # different instruments, one of each
            for name in [x for x in names if part_lookup(x) is not None][:n]:
                out.append((L(en, ru), name, 1, why))
            continue
        name = next((x for x in names if part_lookup(x) is not None), None)
        if name is None:
            continue
        out.append((L(en, ru), name, n, why))
    return out
