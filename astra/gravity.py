"""Gravity assists: fly past a third planet to save fuel.

For a trip A → B around the Sun, every other planet F is tried as a
flyby A → F → B. Both legs are exact Lambert arcs between the planets'
real positions (the same ephemeris the navigator was trained on), over a
grid of departure dates and flight times — thousands of trips at once.

At the flyby the planet turns the craft's velocity for free. How far it
can turn depends on how close we pass:

    δmax = 2·asin(1 / (1 + rp·v∞² / μ))       rp ≥ radius + atmosphere + margin

If the needed turn fits and the speeds in and out match, the flyby costs
nothing; a mismatch is paid with a burn at periapsis (powered flyby,
Oberth effect included). The best trip wins only if it beats the direct
transfer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .i18n import L

CANDIDATES = ("Moho", "Eve", "Kerbin", "Duna", "Dres", "Jool", "Eeloo")
DAY = 21600.0


@dataclass
class Assist:
    flyby: list[str]
    total: float            # Δv from low orbit at A to low orbit at B, m/s
    departure: float
    flyby_dv: float
    arrival: float
    tof: float
    t_dep: float
    t_flyby: float
    t_arr: float
    rp: float               # flyby periapsis altitude, m
    v_inf_arr: float
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        return L(f"{' → '.join(self.flyby)} flyby at {self.rp / 1000:.0f} km, powered part {self.flyby_dv:.0f} m/s",
                 f"пролёт {' → '.join(self.flyby)} на высоте {self.rp / 1000:.0f} км, импульс при пролёте {self.flyby_dv:.0f} м/с")


def _sys():
    from .planets import system
    return system()


def _burn(body, alt, vinf):
    r = body.radius + alt
    return np.sqrt(vinf ** 2 + 2 * body.mu / r) - math.sqrt(body.mu / r)


def _min_rp(body) -> float:
    return body.radius + max(body.atmosphere, body.terrain) + max(10_000.0, body.radius * 0.05)


def search(a: str, b: str, f: str, ut: float, heat: bool = False,
           n_dep: int = 48, n_tof: int = 14) -> Assist | None:
    from kia_core.neural.solar import lambert
    s = _sys()
    A, B, F, sun = s[a], s[b], s[f], s.sun
    mu = sun.mu

    def hoh(x, y):
        aa = (x.sma + y.sma) / 2
        return math.pi * math.sqrt(aa ** 3 / mu)

    syn = abs(1 / (1 / A.period - 1 / B.period))
    t0 = ut + np.linspace(0, syn * 1.2, n_dep)
    k1 = np.linspace(0.35, 1.6, n_tof) * hoh(A, F)
    k2 = np.linspace(0.35, 1.6, n_tof) * hoh(F, B)
    T0, K1, K2 = np.meshgrid(t0, k1, k2, indexing="ij")
    T0, K1, K2 = T0.ravel(), K1.ravel(), K2.ravel()
    T1, T2 = T0 + K1, T0 + K1 + K2
    ra, va = s.state(a, T0)
    rf, vf = s.state(f, T1)
    rb, vb = s.state(b, T2)
    v1d, v1a, ok1 = lambert(ra, rf, K1, mu)
    v2d, v2a, ok2 = lambert(rf, rb, K2, mu)
    vin = v1a - vf
    vout = v2d - vf
    nin, nout = np.linalg.norm(vin, axis=1), np.linalg.norm(vout, axis=1)
    turn = np.arccos(np.clip(np.sum(vin * vout, 1) / np.maximum(nin * nout, 1e-9), -1, 1))
    rp_min = _min_rp(F)
    vbar = (nin + nout) / 2
    dmax = 2 * np.arcsin(1 / (1 + rp_min * vbar ** 2 / F.mu))
    # closest pass that still gives the needed turn (farther = safer)
    need = np.clip(np.sin(turn / 2), 1e-6, 1)
    rp = np.maximum(rp_min, (1 / need - 1) * F.mu / np.maximum(vbar ** 2, 1e-9))
    fb = np.abs(np.sqrt(nout ** 2 + 2 * F.mu / rp) - np.sqrt(nin ** 2 + 2 * F.mu / rp))
    vdep = np.linalg.norm(v1d - va, axis=1)
    varr = np.linalg.norm(v2a - vb, axis=1)
    dep = _burn(A, A.low_orbit(), vdep)
    arr = np.full_like(varr, 80.0) if (heat and B.atmosphere > 0) else _burn(B, B.low_orbit(), varr)
    cost = dep + fb + arr
    bad = ~(ok1 & ok2) | (turn > dmax) | ~np.isfinite(cost)
    cost = np.where(bad, np.inf, cost)
    i = int(np.argmin(cost))
    if not math.isfinite(cost[i]):
        return None
    return Assist([f], float(cost[i]), float(dep[i]), float(fb[i]), float(arr[i]),
                  float(T2[i] - T0[i]), float(T0[i]), float(T1[i]), float(T2[i]),
                  float(rp[i] - F.radius), float(varr[i]))


def direct(a: str, b: str, ut: float, heat: bool = False, n_dep: int = 64, n_tof: int = 24) -> Assist | None:
    """The best direct Lambert transfer over the same horizon — the bar to beat."""
    from kia_core.neural.solar import lambert
    s = _sys()
    A, B = s[a], s[b]
    mu = s.sun.mu
    syn = abs(1 / (1 / A.period - 1 / B.period))
    aa = (A.sma + B.sma) / 2
    hoh = math.pi * math.sqrt(aa ** 3 / mu)
    T0, K = np.meshgrid(ut + np.linspace(0, syn * 1.2, n_dep), np.linspace(0.5, 1.5, n_tof) * hoh, indexing="ij")
    T0, K = T0.ravel(), K.ravel()
    ra, va = s.state(a, T0)
    rb, vb = s.state(b, T0 + K)
    v1, v2, ok = lambert(ra, rb, K, mu)
    dep = _burn(A, A.low_orbit(), np.linalg.norm(v1 - va, axis=1))
    varr = np.linalg.norm(v2 - vb, axis=1)
    arr = np.full_like(varr, 80.0) if (heat and B.atmosphere > 0) else _burn(B, B.low_orbit(), varr)
    cost = np.where(ok, dep + arr, np.inf)
    i = int(np.argmin(cost))
    if not math.isfinite(cost[i]):
        return None
    return Assist([], float(cost[i]), float(dep[i]), 0.0, float(arr[i]), float(K[i]),
                  float(T0[i]), float(T0[i]), float(T0[i] + K[i]), 0.0, float(varr[i]))


_CACHE: dict = {}


def best_assist(a: str, b: str, ut: float, heat: bool = False) -> Assist | None:
    """Best single flyby for A → B, or None when flying direct is cheaper."""
    key = (a, b, round(ut / DAY), heat)
    if key in _CACHE:
        return _CACHE[key]
    found = _from_atlas(a, b, ut, heat)
    if found is not False:
        _CACHE[key] = found
        return found
    base = direct(a, b, ut, heat)
    best = None
    for f in CANDIDATES:
        if f in (a, b):
            continue
        try:
            r = search(a, b, f, ut, heat)
        except Exception:
            continue
        if r is not None and (best is None or r.total < best.total):
            best = r
    if best is not None and base is not None and best.total + 50.0 >= base.total:
        best = None
    if best is not None and base is not None:
        best.extra["saves"] = base.total - best.total
    _CACHE[key] = best
    return best


def _from_atlas(a: str, b: str, ut: float, heat: bool):
    """Best tabulated assist departing in the next window, None if the atlas
    says fly direct, False if the atlas does not cover this date."""
    try:
        from .knowledge import assist_atlas
        atlas = assist_atlas()
    except Exception:
        return False
    rows = atlas.get("pairs", {}).get(f"{a}>{b}")
    if rows is None:
        return False
    s = _sys()
    A, B = s[a], s[b]
    syn = abs(1 / (1 / A.period - 1 / B.period))
    last = float(atlas.get("years", 0.0)) * 426 * DAY
    if ut > last:
        return False                               # beyond the atlas: search live
    near = [r for r in rows if ut <= r["t_dep"] <= ut + syn * 1.2]
    if not near:
        return None
    r = min(near, key=lambda r: r["total"])
    arrival = r["arrival"]
    total = r["total"]
    if heat and B.atmosphere > 0:                  # the atlas assumes a propulsive capture
        total, arrival = total - arrival + 80.0, 80.0
    x = Assist(list(r["flyby"]), total, r["departure"], r["flyby_dv"], arrival,
               r["t_arr"] - r["t_dep"], r["t_dep"], r["t_flyby"], r["t_arr"], r["rp"], r["v_inf_arr"])
    if r.get("direct"):
        x.extra["saves"] = r["direct"] - r["total"]
    return x
