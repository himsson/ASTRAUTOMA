"""Interplanetary targets: every planet of the system, three goals each.

    <PLANET>-L    landing          (not for gas giants)
    <PLANET>-LO   low orbit
    <PLANET>-HO   high orbit

The Δv budget comes from the same exact planner the navigator learned
from: the best departure window after the current game time (Lambert over
a grid of departure times and flight times), capture or aerocapture,
the move to the working orbit and the landing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .i18n import L

GOALS = ("L", "LO", "HO")
OBJECTIVE_OF = {"L": "land", "LO": "low", "HO": "high"}
REAL = {"Moho": ("Mercury", "Меркурий"), "Eve": ("Venus", "Венера"), "Duna": ("Mars", "Марс"),
        "Dres": ("dwarf planet", "карликовая"), "Jool": ("Jupiter", "Юпитер"),
        "Eeloo": ("Pluto", "Плутон")}


def _nav():
    """The navigator's simulator modules (shipped with the app, no training code)."""
    from kia_core.neural import nav_env, solar
    return nav_env, solar


def planet_names() -> list[str]:
    try:
        _, solar = _nav()
        return list(solar.PLANETS)
    except Exception:
        return []


_SYSTEM = None
_TARGETS: dict = {}


def system():
    """Solar system data, read from disk once."""
    global _SYSTEM
    if _SYSTEM is None:
        _, solar = _nav()
        _SYSTEM = solar.SolarSystem()
    return _SYSTEM


def targets() -> list:
    from .i18n import LANG
    if LANG in _TARGETS:
        return list(_TARGETS[LANG])
    _TARGETS[LANG] = _build_targets()
    return list(_TARGETS[LANG])


def _build_targets() -> list:
    from .mission import Target
    try:
        _, solar = _nav()
        system_ = system()
    except Exception:
        return []
    out = []
    for name in solar.PLANETS:
        body = system_[name]
        real = REAL.get(name, ("", ""))
        hint = f" ({L(real[0], real[1])})" if real[0] else ""
        for goal in GOALS:
            if goal == "L" and name in solar.GAS_GIANTS:
                continue
            title = {"L": L(f"{name}{hint} — landing", f"{name}{hint} — посадка"),
                     "LO": L(f"{name}{hint} — low orbit", f"{name}{hint} — низкая орбита"),
                     "HO": L(f"{name}{hint} — high orbit", f"{name}{hint} — высокая орбита")}[goal]
            alt = 0.0 if goal == "L" else (body.low_orbit() if goal == "LO" else body.high_orbit())
            t = Target(f"{name.upper()}-{goal}", title, goal, name, alt, landing=goal == "L")
            t.planet = True
            t.objective = OBJECTIVE_OF[goal]
            out.append(t)
    return out


def current_ut(world) -> float:
    try:
        return float(world.connection.space_center.ut)
    except Exception:
        return float(getattr(world.settings, "ut", 0.0) or 0.0)


@dataclass
class PlanetPlan:
    wait: float           # s until departure
    tof: float            # s of flight
    ejection: float       # Δv from low home orbit
    capture: float        # Δv at arrival to the working orbit
    landing: float        # Δv from low orbit to the ground (0 if not landing)
    aero: bool            # aerocapture chosen
    cap_alt: float        # m, capture periapsis altitude
    v_inf: float
    phase: float          # departure phase angle, degrees


def plan(world, target, heatshield: bool = False, chutes: bool = False) -> PlanetPlan:
    """Exact plan for the next window (same maths the navigator was trained on)."""
    nav_env, solar = _nav()
    env = nav_env.NavigatorEnv(batch=1, stage=3)
    import numpy as np
    ut = current_ut(world)
    obj = nav_env.OBJECTIVES.index(target.objective)
    ctx = {"target": np.array([solar.PLANETS.index(target.body)]), "objective": np.array([obj]),
           "ut0": np.array([ut]), "heatshield": np.array([heatshield]), "chutes": np.array([chutes]),
           "commnet": np.array([False]), "require": np.array([False]), "range_mod": np.array([1.0]),
           "dsn": np.array([3]), "antenna": np.array([1e11]), "relays_there": np.array([0]),
           "dv": np.array([1e5])}
    env.ctx = ctx
    e = env.expert_plan()
    names = [target.body]
    # split arrival Δv: capture (to the working orbit) and landing
    arr_orbit, _ = env._arrival(names, np.array([1 if target.landing else obj]), e["vinf"],
                                e["cap_alt"], e["aero"], ctx["heatshield"], ctx["chutes"])
    arr_full, _ = env._arrival(names, np.array([obj]), e["vinf"], e["cap_alt"], e["aero"],
                               ctx["heatshield"], ctx["chutes"])
    syn = env.synodic[target.body]
    hoh_tof = env.hohmann[target.body][2]
    return PlanetPlan(wait=float(e["wait"][0] * syn), tof=float(e["tof"][0] * hoh_tof),
                      ejection=float(e["ejection"][0]), capture=float(arr_orbit[0]),
                      landing=float(max(0.0, arr_full[0] - arr_orbit[0])),
                      aero=bool(e["aero"][0]), cap_alt=float(e["cap_alt"][0]),
                      v_inf=float(e["vinf"][0]), phase=math.degrees(float(e["phi"][0])))


def budget_legs(world, target, heatshield: bool = False, chutes: bool = False):
    """Maneuvers after low home orbit, with the mission leg keys the rest of the app uses."""
    from .mission import Maneuver
    p = plan(world, target, heatshield, chutes)
    _, solar = _nav()
    body = system()[target.body]
    home = world.home
    legs = [
        Maneuver("tli", L(f"Ejection burn to {target.body}", f"Импульс ухода к {target.body}"),
                 p.ejection, home, twr_min=0.2,
                 note=L(f"window in {p.wait / 21600:.0f} days, φ = {p.phase:.1f}°",
                        f"окно через {p.wait / 21600:.0f} сут, φ = {p.phase:.1f}°")),
        Maneuver("mcc", L("Course corrections (reserve)", "Коррекции курса (резерв)"),
                 0.03 * p.ejection + 30.0, home),
        Maneuver("loi", (L(f"Aerocapture at {target.body} + circularize",
                           f"Аэрозахват у {target.body} + скругление") if p.aero
                         else L(f"Capture at {target.body}", f"Торможение у {target.body}")),
                 p.capture, target.body, twr_min=0.2),
    ]
    if target.landing:
        legs.append(Maneuver("land", L(f"Landing on {target.body}", f"Посадка на {target.body}"),
                             p.landing, target.body, twr_min=2.0 if body.atmosphere == 0 else 1.2,
                             note=L("parachutes do most of the work" if chutes and body.atmosphere
                                    else "powered descent",
                                    "основное делают парашюты" if chutes and body.atmosphere
                                    else "торможение двигателем")))
    duration = p.wait + p.tof + 6 * 3600      # wait + cruise + a few hours at the target
    return legs, duration, p
