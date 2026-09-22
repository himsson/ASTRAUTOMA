"""Rendezvous and docking with another craft (kRPC).

    targets()      every other craft in orbit, grouped by the body they orbit
    Rendezvous     plane match → intercept → kill relative speed → approach
    Docking        RCS: hold the ports face to face, null the side offset,
                   close in slowly until the ports latch

The intercept is searched numerically with the game's own orbit
prediction: burn nodes are tried over a grid of times and Δv, and the one
with the smallest closest-approach distance wins. That works the same for
a craft on a higher, lower or almost identical orbit.

Docking needs RCS thrusters and one free docking port on each craft. The
RCS directions are calibrated at the start (a short pulse on each axis),
so the pilot does not depend on how the craft is built.
"""
from __future__ import annotations

import math
import time

from ..logging_setup import get_logger

log = get_logger("pilot.rendezvous")


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def targets(conn) -> dict[str, list]:
    """{body name: [vessels]} — every other craft that is in orbit or sub-orbit."""
    sc = conn.space_center
    me = sc.active_vessel
    out: dict[str, list] = {}
    for v in sc.vessels:
        try:
            if v == me:
                continue
            sit = str(v.situation).split(".")[-1]
            if sit not in ("orbiting", "sub_orbital", "escaping", "docked"):
                continue
            if "debris" in str(v.type).lower() or "flag" in str(v.type).lower():
                continue
            out.setdefault(v.orbit.body.name, []).append(v)
        except Exception:
            continue
    return out


class Rendezvous:
    APPROACH_DIST = 100.0          # m, where the approach ends
    DOCK_START = 40.0              # m, where docking takes over

    def __init__(self, connection, telemetry, control, maneuver, target):
        self.conn = connection
        self.sc = connection.space_center
        self.tel = telemetry
        self.ctl = control
        self.man = maneuver
        self.target = target

    @property
    def v(self):
        return self.tel.vessel

    # --- geometry ------------------------------------------------------------
    def distance(self) -> float:
        return _norm(self.target.position(self.v.reference_frame))

    def rel_velocity(self):
        """Target's velocity relative to us, in our orbital frame."""
        return self.target.velocity(self.v.orbital_reference_frame)

    def rel_position(self):
        return self.target.position(self.v.orbital_reference_frame)

    # --- 1. planes -------------------------------------------------------------
    def match_planes(self) -> bool:
        o, t = self.v.orbit, self.target.orbit
        inc = o.relative_inclination(t)
        if math.degrees(inc) < 0.1:
            return True
        best = None
        for ta in (o.true_anomaly_at_an(t), o.true_anomaly_at_dn(t)):
            ut = o.ut_at_true_anomaly(ta)
            while ut < self.sc.ut + 60:
                ut += o.period
            r = o.radius_at(ut) if hasattr(o, "radius_at") else o.semi_major_axis
            v = math.sqrt(o.body.gravitational_parameter * (2 / r - 1 / o.semi_major_axis))
            dv = 2 * v * math.sin(inc / 2)
            for sign in (1, -1):
                self.man.clear_nodes()
                node = self.man.add_node(ut, normal=sign * dv)
                left = node.orbit.relative_inclination(t)
                if best is None or left < best[0]:
                    best = (left, ut, sign * dv)
        self.man.clear_nodes()
        node = self.man.add_node(best[1], normal=best[2])
        log.info("Совмещение плоскостей: %.2f° → %.2f°, Δv %.1f м/с",
                 math.degrees(inc), math.degrees(best[0]), abs(best[2]))
        return self.man.execute(node)

    # --- 2. intercept --------------------------------------------------------------
    def _approach_of(self, node) -> float:
        try:
            return node.orbit.distance_at_closest_approach(self.target.orbit)
        except Exception:
            return 1e12

    def intercept(self) -> bool:
        o, t = self.v.orbit, self.target.orbit
        mu = o.body.gravitational_parameter
        r1, r2 = o.semi_major_axis, t.semi_major_axis
        a = (r1 + r2) / 2
        hoh = math.sqrt(mu * (2 / r1 - 1 / a)) - math.sqrt(mu / r1)
        span = max(abs(hoh) * 1.6, 40.0)
        horizon = min(max(o.period, t.period) * 3, abs(1 / (1 / o.period - 1 / t.period)) if o.period != t.period else o.period * 3)
        best = (1e13, None, 0.0)
        now = self.sc.ut + 120.0
        for k in range(60):
            ut = now + horizon * k / 60
            for j in range(-6, 7):
                dv = hoh + span * j / 6
                self.man.clear_nodes()
                d = self._approach_of(self.man.add_node(ut, prograde=dv))
                if d < best[0]:
                    best = (d, ut, dv)
        ut, dv = best[1], best[2]
        st_t, st_v = horizon / 120, span / 12
        for _ in range(30):
            better = False
            for dt, ddv in ((st_t, 0), (-st_t, 0), (0, st_v), (0, -st_v)):
                self.man.clear_nodes()
                d = self._approach_of(self.man.add_node(ut + dt, prograde=dv + ddv))
                if d < best[0]:
                    best, ut, dv, better = (d, ut + dt, dv + ddv), ut + dt, dv + ddv, True
            if not better:
                st_t, st_v = st_t / 2, st_v / 2
            if best[0] < 1_000.0:
                break
        self.man.clear_nodes()
        log.info("Перехват: Δv %.1f м/с, сближение до %.1f км", dv, best[0] / 1000)
        if best[0] > 50_000.0:
            log.warning("Перехват не найден: лучшее сближение %.0f км", best[0] / 1000)
            return False
        return self.man.execute(self.man.add_node(ut, prograde=dv))

    # --- 3. closest approach: stop -------------------------------------------------
    def kill_relative(self, tol: float = 0.5, timeout: float = 240.0) -> bool:
        frame = self.v.orbital_reference_frame
        self.ctl.engage_autopilot(frame)
        deadline = time.time() + timeout
        while time.time() < deadline:
            rv = self.rel_velocity()
            speed = _norm(rv)
            if speed < tol:
                self.ctl.set_throttle(0.0)
                return True
            # our velocity relative to the target is -rv: burn along rv to cancel it
            self.ctl.point_at_vector(_scale(rv, 1 / speed), frame)
            aligned = self.ctl.wait_for_orientation(tolerance=4.0, timeout=30.0)
            acc = max(self.v.available_thrust / max(self.v.mass, 1.0), 0.01)
            self.ctl.set_throttle(min(1.0, speed / acc / 2) if aligned else 0.0)
            time.sleep(0.1)
        self.ctl.set_throttle(0.0)
        return False

    def to_closest_approach(self) -> bool:
        t = self.v.orbit.time_of_closest_approach(self.target.orbit)
        acc = max(self.v.available_thrust / max(self.v.mass, 1.0), 0.01)
        lead = _norm(self.rel_velocity()) / acc / 2 + 30.0
        if t - lead > self.sc.ut + 10:
            self.ctl.warp_to(t - lead)
        return self.kill_relative()

    # --- 4. approach ---------------------------------------------------------------------
    def approach(self, stop: float | None = None, timeout: float = 1800.0) -> bool:
        stop = stop or self.APPROACH_DIST
        frame = self.v.orbital_reference_frame
        self.ctl.engage_autopilot(frame)
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = self.distance()
            if d <= stop:
                return self.kill_relative(0.2)
            pos = self.rel_position()
            dirn = _scale(pos, 1 / max(d, 1e-6))
            want = min(max(d / 60.0, 1.0), 40.0)                 # closing speed for this distance
            # velocity error: we want to move toward the target at `want`
            my_v = _scale(self.rel_velocity(), -1)               # our velocity relative to the target
            err = _sub(_scale(dirn, want), my_v)
            e = _norm(err)
            if e < max(0.5, want * 0.15):
                self.ctl.set_throttle(0.0)
                time.sleep(min(10.0, d / max(want, 1) / 4))
                continue
            self.ctl.point_at_vector(_scale(err, 1 / e), frame)
            if self.ctl.wait_for_orientation(tolerance=5.0, timeout=30.0):
                acc = max(self.v.available_thrust / max(self.v.mass, 1.0), 0.01)
                self.ctl.set_throttle(min(1.0, e / acc / 1.5))
                time.sleep(max(0.2, min(2.0, e / acc)))
            self.ctl.set_throttle(0.0)
        return False

    def run(self, dock: bool = False) -> bool:
        return (self.match_planes() and self.intercept() and self.to_closest_approach()
                and self.approach(self.DOCK_START if dock else self.APPROACH_DIST))


class Docking:
    """Final metres with RCS, port to port."""

    def __init__(self, connection, telemetry, control, target):
        self.sc = connection.space_center
        self.tel = telemetry
        self.ctl = control
        self.target = target
        self.axes = None

    @property
    def v(self):
        return self.tel.vessel

    @staticmethod
    def _ready(ports):
        return [p for p in ports if "ready" in str(p.state).lower()]

    def pick_ports(self):
        mine = self._ready(self.v.parts.docking_ports)
        theirs = self._ready(self.target.parts.docking_ports)
        if not mine or not theirs:
            return None, None
        ref = self.v.reference_frame
        theirs.sort(key=lambda p: _norm(p.position(ref)))
        return mine[0], theirs[0]

    def _command(self, want) -> None:
        """want[i] — our wanted acceleration along axis i of our port frame (−1…1)."""
        c = self.v.control
        for axis, (idx, sign) in self.axes.items():
            setattr(c, axis, max(-1.0, min(1.0, want[idx] * sign)))

    def calibrate(self, port) -> None:
        """Pulses each RCS control axis and finds which way (and along which
        axis of our port frame) it pushes us. Works for any build."""
        c = self.v.control
        frame = port.reference_frame
        self.axes = {}
        for axis in ("right", "forward", "up"):
            v0 = self.target.velocity(frame)
            setattr(c, axis, 1.0)
            time.sleep(0.6)
            setattr(c, axis, 0.0)
            time.sleep(0.3)
            v1 = self.target.velocity(frame)
            ours = [-(v1[k] - v0[k]) for k in range(3)]      # our change = minus the target's
            idx = max(range(3), key=lambda k: abs(ours[k]))
            self.axes[axis] = (idx, 1.0 if ours[idx] >= 0 else -1.0)
            setattr(c, axis, -1.0)                           # undo the pulse
            time.sleep(0.6)
            setattr(c, axis, 0.0)
        log.info("Калибровка RCS: %s", self.axes)

    def run(self, timeout: float = 1200.0) -> bool:
        mine, theirs = self.pick_ports()
        if mine is None:
            log.warning("Нет свободного стыковочного узла")
            return False
        self.v.parts.controlling = mine.part
        try:
            self.sc.target_docking_port = theirs
        except Exception:
            pass
        self.ctl.set_rcs(True)
        frame = mine.reference_frame
        orb = self.v.orbital_reference_frame
        self.ctl.engage_autopilot(orb)
        self.calibrate(mine)
        deadline = time.time() + timeout
        k, kv = 0.08, 1.5
        while time.time() < deadline:
            st = str(mine.state).lower()
            if "docked" in st or "docking" in st:
                self._command((0, 0, 0))
                log.info("Стыковка выполнена")
                return True
            d = theirs.direction(orb)                        # face their port
            self.ctl.point_at_vector((-d[0], -d[1], -d[2]), orb)
            p = theirs.position(frame)                       # x, z — side offset; y — ahead
            vel = theirs.velocity(frame)                     # their motion relative to us
            side = math.hypot(p[0], p[2])
            hold = 0.0 if side < 0.3 else 6.0                # wait 6 m out until lined up
            u = (k * p[0], max(-0.5, min(0.5, k * (p[1] - hold))), k * p[2])
            ours = (-vel[0], -vel[1], -vel[2])
            self._command(tuple(kv * (u[i] - ours[i]) for i in range(3)))
            time.sleep(0.1)
        self._command((0, 0, 0))
        return False
