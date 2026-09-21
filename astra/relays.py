"""Comms relays: how many are needed, how many already fly, where the next goes.

The navigator decides whether a mission has a link home and how many
relay satellites must be delivered first (and to which orbit). Relays are
delivered one per flight: each goes to the relay orbit and is phased into
its own slot of the constellation (360° / N apart) by the flight engine's
constellation pilot.
"""
from __future__ import annotations

from dataclasses import dataclass

from .i18n import L

ORBIT_NAMES = {0: ("low", "низкая"), 1: ("medium", "средняя"), 2: ("high", "высокая")}


@dataclass
class RelayNeed:
    needed: int                 # how many must fly for a reliable link
    present: int                # already in orbit of the target
    orbit_class: int            # 0 low / 1 medium / 2 high
    abort: bool = False
    reason: str = ""

    @property
    def missing(self) -> int:
        return max(0, self.needed - self.present)

    @property
    def orbit_name(self) -> str:
        en, ru = ORBIT_NAMES.get(self.orbit_class, ("high", "высокая"))
        return L(en, ru)


def relay_altitude(body, orbit_class: int) -> float:
    """Relay orbit altitude for a body (BodyInfo): the same classes the navigator learned."""
    if orbit_class == 0:
        return body.radius * 0.6
    if orbit_class == 1:
        return body.radius * 3.0
    return round(min(body.soi * 0.2, body.radius * 12) / 10_000.0) * 10_000.0


def count_present(world, body_name: str) -> int:
    """Relay satellites already orbiting the body (live game only)."""
    conn = world.connection
    if conn is None:
        return 0
    from .craft import _lookup
    n = 0
    try:
        for v in conn.space_center.vessels:
            try:
                if v.orbit.body.name != body_name or "orbiting" not in str(v.situation).lower():
                    continue
                for a in v.parts.antennas:
                    info = _lookup(a.part.name)
                    if info is not None and info.is_relay:
                        n += 1
                        break
            except Exception:
                continue
    except Exception:
        return 0
    return n


def assess(world, target, vessel, dv_available: float) -> RelayNeed | None:
    """Ask the trained navigator about the link for a planet mission."""
    if not getattr(target, "planet", False):
        return None
    try:
        from kia_core.neural.nav_plan import plan_mission
    except Exception:
        return None
    st = world.settings
    present = count_present(world, target.body)
    heat = bool(vessel and any(p.resources_max.get("Ablator", 0) > 0 for p in vessel.parts))
    chutes = bool(vessel and vessel.count("parachute") > 0)
    antenna = max(5_000.0, vessel.best_antenna if vessel else 5_000.0)
    from .planets import current_ut
    try:
        plan = plan_mission(target.body, target.objective, dv_available, ut=current_ut(world),
                            heatshield=heat, chutes=chutes, antenna=antenna, commnet=st.commnet,
                            require=st.require_signal, range_mod=st.range_modifier,
                            dsn=st.tracking_level, relays_there=present)
    except FileNotFoundError:
        return None
    cls = {"низкая": 0, "средняя": 1, "высокая": 2}.get(plan["relay_orbit"], 2)
    need = plan["expert_dv"]
    if plan["abort"] and dv_available < need:
        reason = L(f"not enough Δv: need ≈{need:.0f} m/s, have {dv_available:.0f}",
                   f"Δv не хватит: нужно ≈{need:.0f} м/с, есть {dv_available:.0f}")
    elif plan["abort"]:
        reason = L("no link possible: the tracking station cannot reach the target even through "
                   "a relay, and control without signal is forbidden — upgrade the DSN",
                   "связь невозможна: станция слежения не достаёт до цели даже через "
                   "ретранслятор, а управление без связи запрещено — улучшите DSN")
    else:
        reason = ""
    return RelayNeed(needed=present + int(plan["relays"]), present=present, orbit_class=cls,
                     abort=bool(plan["abort"]), reason=reason)


def relay_target(world, target, slot: int, total: int, orbit_class: int):
    """A copy of the target that ends in the relay orbit instead of the goal."""
    import copy
    from . import mission as ms
    t = copy.copy(target)
    body = ms.target_body(world, target)
    t.altitude = relay_altitude(body, orbit_class)
    t.landing = False
    t.objective = "high"
    t.code = f"{target.body.upper()}-R{slot + 1}"
    t.title = L(f"Relay {slot + 1} of {total} at {target.body}",
                f"Ретранслятор {slot + 1} из {total} у {target.body}")
    t.relay_slot = (slot, total)
    return t


def is_relay(vessel) -> bool:
    return bool(vessel) and vessel.count("relay") > 0
