"""Routes across the whole system: any body → any body, moons included.

The system is a tree (Sun → planets → moons). A trip from A to B climbs
from A up to the common parent, crosses to the sibling branch and climbs
down to B. Every hop is one of five kinds:

    ascend      surface → low orbit of the same body
    up          low orbit of a moon → low orbit of its parent
    across      low orbit of a body → low orbit of a sibling (same parent)
    down        low orbit of a parent → low orbit of one of its moons
    land        low orbit → surface (parachutes where there is air)

Δv of each hop is patched-conic: the hyperbolic excess speed of the
Hohmann ellipse between the two orbits around the common parent, turned
into a burn from the low parking orbit (Oberth effect included). Where
the arrival body has air and the craft has a heat shield, capture is
done by aerobraking and costs only a trim burn.

Returning home is just a route to Kerbin that ends in the atmosphere:
entry, heat shield first, parachutes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .i18n import L

HOME = "Kerbin"
SUN = "Sun"

# Ascent from the surface to low orbit, m/s, for bodies with air (losses included).
# Airless bodies are computed: circular speed plus gravity losses of a steep climb.
ASCENT_ATM = {"Kerbin": 3400.0, "Eve": 8000.0, "Duna": 1450.0, "Laythe": 2900.0}
# Powered part of a landing with parachutes, m/s (thin air needs a burn near the ground)
CHUTE_TRIM = {"Kerbin": 0.0, "Eve": 0.0, "Duna": 250.0, "Laythe": 80.0, "Jool": 0.0}


def system():
    from .planets import system as _s
    return _s()


def moons() -> list[str]:
    s = system()
    return [n for n, b in s.bodies.items() if b.parent not in (None, SUN)]


def chain(name: str) -> list[str]:
    """[name, parent, grandparent, …, Sun]."""
    s = system()
    out = [name]
    while s[out[-1]].parent:
        out.append(s[out[-1]].parent)
    return out


def low_orbit(name: str) -> float:
    return system()[name].low_orbit()


def v_circ(name: str, alt: float) -> float:
    b = system()[name]
    return math.sqrt(b.mu / (b.radius + alt))


def burn_from_orbit(name: str, alt: float, v_inf: float) -> float:
    """Δv to leave (or be captured into) a circular orbit with excess speed v_inf."""
    b = system()[name]
    r = b.radius + alt
    return math.sqrt(v_inf ** 2 + 2 * b.mu / r) - math.sqrt(b.mu / r)


def hohmann_vinf(parent: str, r1: float, r2: float) -> tuple[float, float, float]:
    """(v∞ at r1, v∞ at r2, time of flight) of a Hohmann ellipse around parent."""
    mu = system()[parent].mu
    a = (r1 + r2) / 2
    v1 = abs(math.sqrt(mu * (2 / r1 - 1 / a)) - math.sqrt(mu / r1))
    v2 = abs(math.sqrt(mu / r2) - math.sqrt(mu * (2 / r2 - 1 / a)))
    return v1, v2, math.pi * math.sqrt(a ** 3 / mu)


def ascent_dv(name: str) -> float:
    if name in ASCENT_ATM:
        return ASCENT_ATM[name]
    b = system()[name]
    vc = v_circ(name, low_orbit(name))
    return vc * 1.08 + 30.0 * math.sqrt(b.g0)          # steep climb, short gravity loss


def landing_dv(name: str, chutes: bool) -> float:
    b = system()[name]
    if b.atmosphere > 0 and chutes:
        return CHUTE_TRIM.get(name, 150.0) + 60.0          # deorbit + trim
    if b.atmosphere > 0:
        return v_circ(name, low_orbit(name)) * 0.6          # the air helps a little
    return v_circ(name, low_orbit(name)) * 1.05 + 40.0     # powered all the way


# ==========================================================================
@dataclass
class Hop:
    kind: str                  # ascend / up / across / down / land / entry
    frm: str
    to: str
    dv: float
    tof: float = 0.0           # game seconds
    aero: bool = False
    note: str = ""
    flyby: list[str] = field(default_factory=list)
    arr: float = 0.0           # part of dv spent at arrival (capture)
    park: float = 0.0          # "up" only: stay near the moon's own orbit (radius, m) instead of a low parking orbit

    @property
    def title(self) -> str:
        t = {
            "ascend": L(f"Lift-off from {self.frm} to low orbit", f"Взлёт с {self.frm} на низкую орбиту"),
            "up": L(f"Leave {self.frm} for {self.to} orbit", f"Уход с {self.frm} на орбиту {self.to}"),
            "across": L(f"Transfer {self.frm} → {self.to}", f"Перелёт {self.frm} → {self.to}"),
            "down": L(f"Transfer to {self.to}", f"Перелёт к {self.to}"),
            "land": L(f"Landing on {self.to}", f"Посадка на {self.to}"),
            "entry": L(f"Entry into {self.to} atmosphere, parachutes", f"Вход в атмосферу {self.to}, парашюты"),
        }[self.kind]
        if self.flyby:
            t += L(" via ", " через ") + ", ".join(self.flyby)
        return t


def route(frm: str, to: str, *, landed: bool = False, land: bool = False,
          heatshield: bool = False, chutes: bool = False, ut: float | None = None,
          assists: bool = True) -> list[Hop]:
    """Hops from `frm` (surface if landed, else its low orbit) to `to`
    (surface if land, else its low orbit)."""
    s = system()
    hops: list[Hop] = []
    if landed:
        hops.append(Hop("ascend", frm, frm, ascent_dv(frm)))
    up_chain, down_chain = chain(frm), chain(to)
    common = next(b for b in up_chain if b in down_chain)
    # climb: frm → … → child of common
    i = 0
    while up_chain[i + 1] != common and up_chain[i] != common:
        moon, parent = up_chain[i], up_chain[i + 1]
        hops.append(_up(moon, parent, heatshield))
        i += 1
    a = up_chain[i]
    j = down_chain.index(common)
    k = j                                           # where the descent starts
    if a != common and j > 0:
        b = down_chain[j - 1]
        if a != b:
            hops.append(_across(a, b, common, heatshield, ut, assists))
        k = j - 1
    elif a != common and j == 0:
        hops.append(_up(a, common, heatshield))     # the target is the parent itself
        k = 0
    # descend: … → to
    while k > 0:
        parent, moon = down_chain[k], down_chain[k - 1]
        hops.append(_down(parent, moon, heatshield))
        k -= 1
    _oberth_fix(hops, heatshield)
    if land:
        if to == HOME and chutes and heatshield:
            last = hops[-1] if hops else None
            if last is not None and last.kind in ("up", "across") and last.to == HOME:
                # straight in from the transfer: no parking orbit at home
                last.dv -= last.arr
                last.arr = 0.0
                last.aero = True
                last.note = L("straight into the atmosphere", "сразу в атмосферу")
            hops.append(Hop("entry", HOME, HOME, 0.0))
        else:
            hops.append(Hop("land", to, to, landing_dv(to, chutes)))
    return hops


def _oberth_fix(hops: list[Hop], heat: bool) -> None:
    """Leaving a moon's system for another planet: do not dive to a low
    parking orbit of the parent and climb out again. Escape the moon, stay
    near its orbit around the parent, and leave from there."""
    s = system()
    for i in range(len(hops) - 1):
        up, nxt = hops[i], hops[i + 1]
        if up.kind != "up" or nxt.kind != "across" or nxt.frm != up.to or nxt.flyby:
            continue
        m, p = s[up.frm], s[up.to]
        esc = burn_from_orbit(up.frm, low_orbit(up.frm), 0.0) + 20.0
        # departure of the next hop from a circular orbit at the moon's radius
        A, B = p, s[nxt.to]
        v1, _, _ = hohmann_vinf(A.parent, A.sma, B.sma)
        r = m.sma * 0.9
        dep_far = math.sqrt(v1 ** 2 + 2 * p.mu / r) - math.sqrt(p.mu / r)
        dep_low = burn_from_orbit(p.name, low_orbit(p.name), v1)
        new_total = esc + dep_far
        old_total = up.dv + dep_low
        if new_total < old_total:
            up.dv, up.arr, up.park, up.aero = esc, 0.0, r, False
            up.note = L(f"escape only, stay at {r / 1e6:.1f} Mm around {up.to}",
                        f"только уход, остаёмся на {r / 1e6:.1f} тыс. км вокруг {up.to}")
            nxt.dv = nxt.dv - dep_low + dep_far


def _up(moon: str, parent: str, heat: bool) -> Hop:
    s = system()
    m, p = s[moon], s[parent]
    v1, v2, tof = hohmann_vinf(parent, m.sma, p.radius + low_orbit(parent))
    dv = burn_from_orbit(moon, low_orbit(moon), v1)
    arr = v2                     # the ellipse ends at the parking orbit: one burn at periapsis
    aero = heat and p.atmosphere > 0
    arr = 60.0 if aero else arr
    h = Hop("up", moon, parent, dv + arr, tof, aero, arr=arr)
    h._vinf_arr = v2
    return h


def _down(parent: str, moon: str, heat: bool) -> Hop:
    s = system()
    m, p = s[moon], s[parent]
    v1, v2, tof = hohmann_vinf(parent, p.radius + low_orbit(parent), m.sma)
    dv = v1                      # a prograde burn from the parking orbit
    aero = heat and m.atmosphere > 0
    arr = 60.0 if aero else burn_from_orbit(moon, low_orbit(moon), v2)
    return Hop("down", parent, moon, dv + arr, tof, aero, arr=arr)


def _across(a: str, b: str, parent: str, heat: bool, ut, assists: bool) -> Hop:
    s = system()
    A, B = s[a], s[b]
    v1, v2, tof = hohmann_vinf(parent, A.sma, B.sma)
    dep = burn_from_orbit(a, low_orbit(a), v1)
    aero = heat and B.atmosphere > 0
    arr = 80.0 if aero else burn_from_orbit(b, low_orbit(b), v2)
    h = Hop("across", a, b, dep + arr, tof, aero, arr=arr)
    h._vinf_arr = v2
    if assists and parent == SUN and _can("assists"):
        from . import gravity
        best = gravity.best_assist(a, b, ut or 0.0, heat)
        if best is not None and best.total + 50.0 < h.dv:
            h = Hop("across", a, b, best.total, best.tof, aero, flyby=best.flyby, arr=best.arrival,
                    note=L(f"gravity assist saves {h.dv - best.total:.0f} m/s",
                           f"гравитационный манёвр экономит {h.dv - best.total:.0f} м/с"))
            h._vinf_arr = best.v_inf_arr
            h._assist = best
    return h


def total(hops: list[Hop]) -> float:
    return sum(h.dv for h in hops)


def duration(hops: list[Hop]) -> float:
    return sum(h.tof for h in hops) + 3600.0 * len(hops)


def where(vessel_body: str, situation: str) -> tuple[str, bool]:
    """(body, landed) of a craft from kRPC's orbit body and situation."""
    return vessel_body, situation in ("landed", "splashed", "pre_launch")


def _can(skill: str) -> bool:
    try:
        from .knowledge import capabilities
        return bool(capabilities().get(skill))
    except Exception:
        return False
