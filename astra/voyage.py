"""Flying a route hop by hop in the live game (kRPC).

Every hop of `routes.route()` becomes one or more steps of the flight
plan. The pilots here are generic — they read the bodies from the game,
so the same code lifts off from the Mun, leaves Ike for Duna, crosses
from Duna to Kerbin and brings the craft down on parachutes.

    ascend   vertical climb, gravity turn, coast, circularize
    up       escape burn from a moon: the burn point is searched around
             the whole orbit with the game's own patched conics until the
             orbit after leaving the moon has the wanted periapsis
    across / down
             the flight engine's transfer planner (window, node, course
             corrections, capture), aerobraking where the plan says so
    land     powered landing (airless) or parachutes (air)
    entry    home: heat shield forward, parachutes, touchdown
"""
from __future__ import annotations

import logging
import math
import time

from .i18n import L

log = logging.getLogger("kia.astra.voyage")


class Voyage:
    def __init__(self, ctx: dict):
        self.c = ctx

    # --- helpers -------------------------------------------------------------
    @property
    def sc(self):
        return self.c["conn"].space_center

    @property
    def v(self):
        return self.c["tel"].vessel

    def body(self, name: str):
        return self.sc.bodies[name]

    def here(self) -> str:
        return self.v.orbit.body.name

    def _low(self, name: str) -> float:
        from .routes import low_orbit
        return low_orbit(name)

    def _planner(self, target: str, periapsis: float):
        from kia_core.pilot.transfer import TransferPlanner
        c = self.c
        tr = TransferPlanner(c["conn"], c["tel"], c["ctl"], c["man"], c["g"], c["rewards"],
                             target=target, target_periapsis=periapsis)
        tr._math_brain = False
        c["transfer"] = tr
        return tr

    # --- ascend: surface → low orbit ----------------------------------------------
    def ascend(self, name: str) -> bool:
        ctl, tel, man = self.c["ctl"], self.c["tel"], self.c["man"]
        b = self.body(name)
        target = self._low(name)
        atm = b.atmosphere_depth if b.has_atmosphere else 0.0
        turn_end = atm * 0.75 if atm else max(8_000.0, target * 0.35)
        start = 250.0 if not atm else 1_000.0
        ctl.set_sas(False)
        ctl.engage_autopilot()
        ctl.point(90.0, 90.0)
        ctl.set_throttle(1.0)
        if self.v.control.current_stage and not self.v.available_thrust:
            ctl.next_stage()
        deadline = time.time() + 900
        while time.time() < deadline:
            ctl.autostage()
            alt = tel.get("altitude")
            ap = self.v.orbit.apoapsis_altitude
            if ap >= target:
                break
            if alt > start:
                frac = min(1.0, (alt - start) / max(turn_end - start, 1.0))
                pitch = 90.0 * (1.0 - math.sqrt(frac))
                ctl.point(max(pitch, 0.0 if not atm else 5.0), 90.0)
            time.sleep(0.1)
        ctl.set_throttle(0.0)
        if atm:                                  # coast out of the air before circularizing
            while tel.get("altitude") < atm and time.time() < deadline:
                ctl.autostage()
                if self.v.orbit.apoapsis_altitude < target * 0.98:
                    ctl.point(0.0, 90.0)
                    ctl.set_throttle(0.3)
                else:
                    ctl.set_throttle(0.0)
                time.sleep(0.2)
        return bool(man.circularize_at_apoapsis(0.5)) and self.v.orbit.periapsis_altitude > atm

    # --- up: moon → orbit of its parent ---------------------------------------------
    def escape_up(self, parent: str, periapsis: float) -> bool:
        """Leaves the current moon so that, around the parent, the periapsis
        is `periapsis` (m). Searches the burn point over one full orbit."""
        man, ctl = self.c["man"], self.c["ctl"]
        moon = self.v.orbit.body
        P = self.body(parent)
        mu_p = P.gravitational_parameter
        r_moon = moon.orbit.semi_major_axis
        r_pe = P.equatorial_radius + periapsis
        a = (r_moon + r_pe) / 2
        v_inf = math.sqrt(mu_p / r_moon) - math.sqrt(mu_p * (2 / r_moon - 1 / a))
        r = self.v.orbit.semi_major_axis
        mu_m = moon.gravitational_parameter
        dv0 = math.sqrt(v_inf ** 2 + 2 * mu_m / r) - math.sqrt(mu_m / r)

        def score(node) -> float:
            try:
                nxt = node.orbit.next_orbit
                if nxt is None or nxt.body.name != parent:
                    return 1e12
                return abs(nxt.periapsis_altitude - periapsis)
            except Exception:
                return 1e12

        period = self.v.orbit.period
        t0 = self.sc.ut + 120.0
        best = (1e13, None, dv0)
        for k in range(48):
            for dv in (dv0, dv0 * 1.05, dv0 * 1.12):
                man.clear_nodes()
                node = man.add_node(t0 + period * k / 48, prograde=dv)
                sc_ = score(node)
                if sc_ < best[0]:
                    best = (sc_, t0 + period * k / 48, dv)
        man.clear_nodes()
        if best[1] is None or best[0] >= 1e12:
            log.warning("Уход с %s: нет выхода к %s", moon.name, parent)
            return False
        ut, dv = best[1], best[2]
        step_t, step_v = period / 96, dv0 * 0.02
        for _ in range(24):                       # coordinate descent on time and Δv
            improved = False
            for dt, ddv in ((step_t, 0), (-step_t, 0), (0, step_v), (0, -step_v)):
                man.clear_nodes()
                node = man.add_node(ut + dt, prograde=dv + ddv)
                s_ = score(node)
                if s_ < best[0]:
                    best, ut, dv, improved = (s_, ut + dt, dv + ddv), ut + dt, dv + ddv, True
            if not improved:
                step_t, step_v = step_t / 2, step_v / 2
            if best[0] < max(1_000.0, periapsis * 0.05):
                break
        man.clear_nodes()
        node = man.add_node(ut, prograde=dv)
        log.info("Уход с %s: Δv %.0f м/с, перицентр у %s будет %.0f км (промах %.0f км)",
                 moon.name, dv, parent, periapsis / 1000, best[0] / 1000)
        if not man.execute(node):
            return False
        for _ in range(4):                        # to the parent's sphere
            if self.here() == parent:
                break
            tts = self.v.orbit.time_to_soi_change
            if not math.isfinite(tts) or tts <= 0:
                break
            ctl.warp_to(self.sc.ut + tts + 30.0)
            time.sleep(1.0)
        self.c["tel"].rebind()
        ctl.rebind()
        return self.here() == parent

    def capture_here(self, altitude: float, aero: bool) -> bool:
        """At periapsis around the current body: aerobrake or burn into a circular orbit."""
        man = self.c["man"]
        b = self.v.orbit.body
        if aero and b.has_atmosphere:
            from .interplanetary import PlanetFlight
            self._planner(b.name, altitude)
            return PlanetFlight(self.c, b.name, altitude, True)._aerocapture()
        ok = man.circularize_at_periapsis(0.5)
        if ok and abs(self.v.orbit.periapsis_altitude - altitude) > altitude * 0.25:
            ok = self._planner(b.name, altitude).lower_to_target()
        return bool(ok)

    def up(self, moon: str, parent: str, aero: bool, entry: bool, park: float = 0.0) -> bool:
        P = self.body(parent)
        if park:                                   # just escape: stay near the moon's orbit
            return self.escape_up(parent, park - P.equatorial_radius)
        if entry:                                  # straight into the home atmosphere
            pe = P.atmosphere_depth * 0.42
        elif aero and P.has_atmosphere:
            pe = P.atmosphere_depth * 0.55
        else:
            pe = self._low(parent)
        if not self.escape_up(parent, pe):
            return False
        if entry:
            return True
        return self.capture_here(self._low(parent), aero)

    # --- across / down: to a sibling or a moon ----------------------------------------
    def transfer(self, target: str, aero: bool, entry: bool = False) -> bool:
        T = self.body(target)
        if entry:
            pe = T.atmosphere_depth * 0.42
        elif aero and T.has_atmosphere:
            pe = T.atmosphere_depth * 0.55
        else:
            pe = self._low(target)
        tr = self._planner(target, pe)
        node = tr.plan_transfer()
        if node is None:
            return False
        if not tr.execute_transfer(node):
            return False
        if self._interplanetary(target):
            from .interplanetary import PlanetFlight
            pf = PlanetFlight(self.c, target, pe, aero)
            pf.depart_ut = self.sc.ut
            try:
                pf.tof = float(tr._transfer_time())
            except Exception:
                pf.tof = None
            if not pf.cruise():
                return False
        else:
            if not tr.coast_to_soi():
                return False
        if entry:
            return True
        if aero and T.has_atmosphere:
            return self.capture_here(self._low(target), True)
        tr.target_periapsis = self._low(target)
        return bool(tr.capture())

    def flyby_transfer(self, hop, entry: bool = False) -> bool:
        """A → F (flyby) → B on the dates the assist planner found."""
        a = getattr(hop, "_assist", None)
        if a is None:
            return self.transfer(hop.to, hop.aero, entry)
        ctl = self.c["ctl"]
        fb = hop.flyby[0]
        tr = self._planner(fb, a.rp)
        # the planner's own window is a Hohmann to F; the assist needs its own date
        tr.time_to_window = lambda: max(0.0, a.t_dep - self.sc.ut)
        tr.injection_delta_v = lambda: a.departure
        node = tr.plan_transfer()
        if node is None or not tr.execute_transfer(node):
            return False
        tr.target_periapsis = a.rp
        for frac in (0.3, 0.8, 0.97):            # aim the flyby periapsis
            when = a.t_dep + (a.t_flyby - a.t_dep) * frac
            if when > self.sc.ut + 60:
                ctl.warp_to(when)
            if self.c["tel"].is_in_soi(fb):
                break
            tr.correct_course()
        if not tr.coast_to_soi(timeout=1800.0):
            return False
        # pass the planet, then steer for the real target
        for _ in range(3):
            if self.here() != fb:
                break
            tts = self.v.orbit.time_to_soi_change
            if not math.isfinite(tts) or tts <= 0:
                break
            ctl.warp_to(self.sc.ut + tts + 60.0)
            time.sleep(1.0)
        self.c["tel"].rebind()
        ctl.rebind()
        T = self.body(hop.to)
        pe = (T.atmosphere_depth * (0.42 if entry else 0.55)) if (hop.aero or entry) and T.has_atmosphere             else self._low(hop.to)
        tr2 = self._planner(hop.to, pe)
        tr2._home_name = fb
        for frac in (0.1, 0.6, 0.95):
            when = a.t_flyby + (a.t_arr - a.t_flyby) * frac
            if when > self.sc.ut + 60:
                ctl.warp_to(when)
            if self.c["tel"].is_in_soi(hop.to):
                break
            tr2.correct_course()
        if not tr2.coast_to_soi(timeout=1800.0):
            return False
        if entry:
            return True
        if hop.aero and T.has_atmosphere:
            return self.capture_here(self._low(hop.to), True)
        tr2.target_periapsis = self._low(hop.to)
        return bool(tr2.capture())

    def _interplanetary(self, target: str) -> bool:
        try:
            return self.body(target).orbit.body.name != self.here()
        except Exception:
            return False

    # --- land / entry --------------------------------------------------------------
    def land(self, name: str) -> bool:
        b = self.body(name)
        if b.has_atmosphere:
            from .interplanetary import ChuteLanding
            ch = ChuteLanding(self.c, name)
            return ch.deorbit() and ch.entry() and ch.touchdown()
        from kia_core.pilot.landing import LandingPilot
        c = self.c
        lp = LandingPilot(c["conn"], c["tel"], c["ctl"], c["man"], c["rewards"], target_name=name)
        return lp.deorbit() and lp.coast_to_surface() and lp.kill_horizontal() and lp.descend()

    def entry(self, name: str) -> bool:
        from .interplanetary import ChuteLanding
        ch = ChuteLanding(self.c, name)
        b = self.body(name)
        try:
            if self.v.orbit.periapsis_altitude > b.atmosphere_depth:
                if not ch.deorbit():
                    return False
        except Exception:
            pass
        return ch.entry() and ch.touchdown()


def steps_for(hops, add, Step) -> None:
    """Adds plan steps for a route. `add(step)` appends to the plan; the
    step's `hop` attribute tells the executor what to fly."""
    entry_next = {i for i, h in enumerate(hops[:-1]) if hops[i + 1].kind == "entry"}
    for i, h in enumerate(hops):
        s = Step(f"v{i}_{h.kind}", h.title, h.note, wall=_wall(h), dv=h.dv)
        s.hop = h
        s.entry_next = i in entry_next
        add(s)


def _wall(h) -> float:
    return {"ascend": 240.0, "up": 180.0, "across": 420.0, "down": 240.0,
            "land": 300.0, "entry": 360.0}.get(h.kind, 120.0)


def runner(voy: Voyage, step):
    h = step.hop
    if h.kind == "ascend":
        return lambda: voy.ascend(h.frm)
    if h.kind == "up":
        return lambda: voy.up(h.frm, h.to, h.aero, step.entry_next, h.park)
    if h.kind == "across" and h.flyby:
        return lambda: voy.flyby_transfer(h, step.entry_next)
    if h.kind in ("across", "down"):
        return lambda: voy.transfer(h.to, h.aero, step.entry_next)
    if h.kind == "land":
        return lambda: voy.land(h.to)
    if h.kind == "entry":
        return lambda: voy.entry(h.to)
    return lambda: True
