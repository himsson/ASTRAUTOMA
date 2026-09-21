"""Flying to a planet with the trained execution profile.

The navigator decides where and when; this module flies it using the
flight engine's pilots (window, ejection, course corrections, capture) and
the parameters the autopilot learned in the cruise simulator:

    eject_lead            share of the ejection burn done before the node
    corr1 / corr2 / corr3 when to correct the course (share of the trip)
    threshold_frac        do not touch the course if the miss is smaller
                          (share of the planet radius)
    cap_bias              capture periapsis relative to the working orbit
    aero_depth            aerocapture periapsis depth (share of atmosphere)
    capture_lead          share of the capture burn before periapsis
"""
from __future__ import annotations

import logging
import math
import time

log = logging.getLogger("kia.astra.interplanetary")

DEFAULTS = {"eject_lead": 0.5, "corr1": 0.5, "corr2": 0.95, "corr3": 0.995, "threshold": 0.3,
            "cap_bias": 0.3, "aero_depth": 0.5, "capture_lead": 0.5}


def profile(planet: str, aero: bool) -> dict:
    """Trained execution parameters for this planet (or safe defaults)."""
    from .knowledge import genome
    block = genome().get("interplanetary", {}).get("planets", {}).get(planet, {})
    p = dict(DEFAULTS)
    p.update(block.get("aero" if aero else "propulsive") or block.get("propulsive") or {})
    p["threshold_frac"] = 10 ** (-3 + 3 * float(p.get("threshold", 0.3)))
    return p


class PlanetFlight:
    """One trip: ejection → cruise with corrections → capture → working orbit."""

    def __init__(self, ctx: dict, planet: str, goal_alt: float, aero: bool):
        self.c = ctx
        self.planet = planet
        self.goal_alt = goal_alt
        self.aero = aero
        self.p = profile(planet, aero)
        self.depart_ut = None
        self.tof = None

    @property
    def tr(self):
        return self.c["transfer"]

    @property
    def sc(self):
        return self.c["conn"].space_center

    def body(self):
        return self.sc.bodies[self.planet]

    # --- window and ejection ----------------------------------------------
    def window(self) -> bool:
        node = self.tr.plan_transfer()          # waits for the window, builds the node
        if node is None:
            return False
        self.c["node"] = node
        return True

    def eject(self) -> bool:
        node = self.c.get("node")
        if node is None:
            return False
        self.c["g"].node_execute_lead = float(self.p["eject_lead"])
        ok = self.tr.execute_transfer(node)
        self.depart_ut = self.sc.ut
        try:
            self.tof = float(self.tr._transfer_time())
        except Exception:
            self.tof = None
        return ok

    # --- cruise ------------------------------------------------------------
    def _capture_alt(self) -> float:
        b = self.body()
        if self.aero and b.has_atmosphere:
            return float(self.p["aero_depth"]) * b.atmosphere_depth
        return self.goal_alt * (1.0 + float(self.p["cap_bias"]))

    def cruise(self) -> bool:
        """Leave home, then correct the course at the learned moments only
        when the predicted periapsis misses by more than the learned threshold."""
        tr, ctl = self.tr, self.c["ctl"]
        tr.target_periapsis = self._capture_alt()
        home = getattr(tr, "_home_name", None) or self.c["home"]
        for _ in range(3):
            try:
                if self.c["tel"].vessel.orbit.body.name != home:
                    break
                tts = self.c["tel"].vessel.orbit.time_to_soi_change
            except Exception:
                break
            if not math.isfinite(tts) or tts <= 0:
                break
            ctl.warp_to(self.sc.ut + tts + 60.0)
        tof = self.tof or 0.0
        start = self.depart_ut or self.sc.ut
        radius = self.body().equatorial_radius
        for key in ("corr1", "corr2", "corr3"):
            if self.c["tel"].is_in_soi(self.planet):
                break
            when = start + tof * float(self.p[key]) if tof else None
            if when and when > self.sc.ut + 60:
                ctl.warp_to(when)
            pe = tr._encounter_periapsis_now()
            miss = abs(pe - tr.target_periapsis) if pe is not None else float("inf")
            if miss > float(self.p["threshold_frac"]) * radius:
                log.info("Коррекция %s: промах %.0f км (порог %.0f км)", key,
                         miss / 1000 if math.isfinite(miss) else -1,
                         float(self.p["threshold_frac"]) * radius / 1000)
                tr.correct_course()
        return tr.coast_to_soi(timeout=1800.0)

    # --- capture -----------------------------------------------------------
    def capture(self) -> bool:
        b = self.body()
        if self.aero and b.has_atmosphere:
            return self._aerocapture()
        tr = self.tr
        tr.target_periapsis = self._capture_alt()
        tr._raise_periapsis(tr.target_periapsis)
        self.c["g"].capture_lead = float(self.p["capture_lead"])
        tr.target_periapsis = self.goal_alt
        return tr.capture()

    def _aerocapture(self) -> bool:
        tr, ctl, man = self.tr, self.c["ctl"], self.c["man"]
        b = self.body()
        depth = float(self.p["aero_depth"]) * b.atmosphere_depth
        tr._raise_periapsis(depth)                         # into the corridor
        v = self.c["tel"].vessel
        ctl.warp_to(self.sc.ut + max(0.0, v.orbit.time_to_periapsis - 180.0))
        ctl.disengage_autopilot(hold=False)
        ctl.set_sas(True, "retrograde")                    # heat shield forward
        deadline = time.time() + 900
        entered = False
        while time.time() < deadline:
            alt = self.c["tel"].get("altitude")
            if alt < b.atmosphere_depth:
                entered = True
            if entered and alt > b.atmosphere_depth * 1.02:
                break
            time.sleep(0.5)
        v = self.c["tel"].vessel
        if v.orbit.apoapsis <= 0 or v.orbit.apoapsis_altitude > b.sphere_of_influence:
            log.warning("Аэрозахват: орбита не замкнулась")
            return False
        # raise periapsis out of the air at apoapsis, then round the orbit
        mu = b.gravitational_parameter
        ra = v.orbit.apoapsis
        rg = b.equatorial_radius + self.goal_alt
        a_new = (ra + rg) / 2.0
        dv = math.sqrt(mu * (2 / ra - 1 / a_new)) - math.sqrt(mu * (2 / ra - 1 / v.orbit.semi_major_axis))
        man.clear_nodes()
        node = man.add_node(self.sc.ut + v.orbit.time_to_apoapsis, prograde=dv)
        if not man.execute(node):
            return False
        return man.circularize_at_periapsis()

    def working_orbit(self) -> bool:
        tr = self.tr
        tr.target_periapsis = self.goal_alt
        return tr.lower_to_target()


class ChuteLanding:
    """Landing through an atmosphere: deorbit into the air, parachutes, a
    short burn near the ground when the air is thin (Duna)."""

    def __init__(self, ctx: dict, planet: str):
        self.c = ctx
        self.planet = planet

    def body(self):
        return self.c["conn"].space_center.bodies[self.planet]

    def deorbit(self) -> bool:
        b = self.body()
        v = self.c["tel"].vessel
        mu = b.gravitational_parameter
        r = v.orbit.apoapsis
        rp = b.equatorial_radius + b.atmosphere_depth * 0.25
        a_new = (r + rp) / 2.0
        dv = math.sqrt(mu * (2 / r - 1 / a_new)) - math.sqrt(mu * (2 / r - 1 / v.orbit.semi_major_axis))
        man = self.c["man"]
        man.clear_nodes()
        node = man.add_node(self.c["conn"].space_center.ut + 60.0, prograde=dv)
        return man.execute(node)

    def entry(self) -> bool:
        b = self.body()
        ctl, tel = self.c["ctl"], self.c["tel"]
        v = tel.vessel
        ctl.warp_to(self.c["conn"].space_center.ut + max(0.0, v.orbit.time_to_periapsis * 0.8))
        ctl.disengage_autopilot(hold=False)
        ctl.set_sas(True, "retrograde")
        deadline = time.time() + 1200
        while time.time() < deadline:
            alt = tel.get("altitude")
            speed = tel.get("speed")
            if alt < b.atmosphere_depth * 0.35 and speed < 450:
                ctl.deploy_parachutes()
                ctl.deploy_landing_gear()
                return True
            time.sleep(0.3)
        return False

    def touchdown(self) -> bool:
        ctl, tel = self.c["ctl"], self.c["tel"]
        deadline = time.time() + 1200
        while time.time() < deadline:
            sit = tel.situation()
            if sit in ("landed", "splashed"):
                ctl.set_throttle(0.0)
                return True
            h = tel.get("surface_altitude")
            vs = tel.get("vertical_speed")
            if h < 300 and vs < -7.0:
                ctl.set_sas(True, "retrograde")
                ctl.set_throttle(min(1.0, max(0.0, (-vs - 4.0) / 6.0)))
            else:
                ctl.set_throttle(0.0)
            time.sleep(0.1)
        return False
