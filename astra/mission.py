"""Цели миссий и бюджет Δv.

Все высоты и импульсы выводятся из параметров тел текущего мира, а не
из таблиц: при другом масштабе системы (RSS, Sigma Dimensions) цифры
пересчитываются сами.

Обозначения:
    Δv   — характеристическая скорость, м/с
    Hπ   — высота перицентра, Hα — высота апоцентра
    φ    — фазовый угол цели
    TWR  — тяговооружённость (тяга / вес в местном g)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .i18n import L
from .world import BodyInfo, World, terrain_estimate

MARGIN = 1.15              # «с запасом»: +15 % к расчётному бюджету
DV_ASCENT_ATM_SHARE = 0.45  # доля выведения, которую первая ступень идёт в плотной атмосфере


@dataclass
class Target:
    code: str
    title: str
    short: str
    body: str             # у какого тела заканчивается полёт
    altitude: float       # высота конечной орбиты (0 — поверхность)
    landing: bool = False
    planet: bool = False       # interplanetary target
    objective: str = ""        # land / low / high for planets

    def label(self, world: World) -> str:
        return f"{self.code} · {self.title}"


@dataclass
class Maneuver:
    key: str
    title: str
    dv: float
    body: str
    atmospheric: bool = False
    twr_min: float = 0.0       # минимальная TWR в местном g
    note: str = ""


@dataclass
class Budget:
    target: Target
    legs: list[Maneuver] = field(default_factory=list)
    duration: float = 0.0      # оценка длительности полёта в игровых секундах

    @property
    def total(self) -> float:
        return sum(l.dv for l in self.legs)

    @property
    def total_margin(self) -> float:
        return self.total * MARGIN


# ==========================================================================
def leo_altitude(home: BodyInfo) -> float:
    if home.atmosphere_depth > 0:
        return math.ceil((home.atmosphere_depth + 10_000.0) / 5_000.0) * 5_000.0
    return max(10_000.0, terrain_estimate(home) * 1.5)


def heo_altitude(home: BodyInfo) -> float:
    """Высокая орбита — синхронная (стационарная), если она внутри SOI."""
    sync = home.synchronous_altitude()
    if math.isfinite(sync) and sync + home.radius < home.soi * 0.5:
        return round(sync / 1000.0) * 1000.0
    return round((home.soi * 0.25 - home.radius) / 10_000.0) * 10_000.0


def llo_altitude(moon: BodyInfo) -> float:
    alt = max(terrain_estimate(moon) + 10_000.0, moon.radius * 0.1)
    return math.ceil(alt / 5_000.0) * 5_000.0


def hlo_altitude(moon: BodyInfo) -> float:
    return round((moon.soi * 0.2) / 10_000.0) * 10_000.0


def targets(world: World) -> list[Target]:
    h, m = world.home_body, world.moon_body
    return [
        Target("LEO", L("Low Earth orbit", "Низкая околоземная орбита"), "LEO", h.name, leo_altitude(h)),
        Target("HEO", L("High Earth orbit", "Высокая околоземная орбита"), "HEO", h.name, heo_altitude(h)),
        Target("LLO", L("Low lunar orbit", "Низкая окололунная орбита"), "LLO", m.name, llo_altitude(m)),
        Target("HLO", L("High lunar orbit", "Высокая окололунная орбита"), "HLO", m.name, hlo_altitude(m)),
        Target("LND", L("Moon landing", "Посадка на Луну"), "LND", m.name, 0.0, landing=True),
    ] + planet_targets()


def planet_targets() -> list:
    from .planets import targets as _pt
    return _pt()


def target_body(world: World, target: Target) -> BodyInfo:
    """The body the mission ends at: home, the Moon or a planet."""
    if getattr(target, "planet", False):
        from .planets import system as SolarSystem
        b = SolarSystem()[target.body]
        return BodyInfo(b.name, b.mu, b.radius, b.rotation_period, b.atmosphere, b.soi, b.parent,
                        b.sma, b.inc, b.lan, b.period, b.terrain)
    if target.body == world.moon:
        return world.moon_body
    return world.home_body


def work_altitude(world: World, target: Target) -> float:
    """Orbit altitude the mission works from (landings start from the low orbit)."""
    if not target.landing:
        return target.altitude
    body = target_body(world, target)
    if getattr(target, "planet", False):
        from .planets import system as SolarSystem
        return SolarSystem()[target.body].low_orbit()
    return llo_altitude(body)


def comm_distance(world: World, target: Target) -> float:
    """Worst distance from home to the craft during the mission, m."""
    h = world.home_body
    if getattr(target, "planet", False):
        from .planets import system as SolarSystem
        s = SolarSystem()
        return s[target.body].sma + s[world.home].sma if world.home in s.bodies else s[target.body].sma * 2
    if target.body == world.moon:
        m = world.moon_body
        return m.sma + m.soi
    return (h.radius + max(target.altitude, leo_altitude(h))) * 2


def target_by_code(world: World, code: str) -> Target:
    for t in targets(world):
        if t.code == code:
            return t
    return targets(world)[0]


# ==========================================================================
# Импульсы
# ==========================================================================
def ascent_dv(home: BodyInfo, altitude: float, latitude_deg: float = 0.0,
              heading_deg: float = 90.0) -> float:
    """Выведение на круговую орбиту с учётом вращения планеты.

    Потери на гравитацию и сопротивление: 1.1·√(2·H_атм·g₀) — для Кербина
    это ≈1290 м/с, итог ≈3400 м/с, как на стандартной карте Δv.
    """
    v = home.v_circ(altitude)
    rot = home.rotation_speed * math.cos(math.radians(latitude_deg)) \
        * math.sin(math.radians(heading_deg))
    if home.atmosphere_depth > 0:
        losses = 1.1 * math.sqrt(2.0 * home.atmosphere_depth * home.g0)
    else:
        losses = 0.08 * v
    return v - rot + losses


def hohmann(body: BodyInfo, r1: float, r2: float) -> tuple[float, float, float]:
    """Два импульса и время перелёта по полуэллипсу Гомана."""
    mu = body.mu
    a = (r1 + r2) / 2.0
    dv1 = math.sqrt(mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = math.sqrt(mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
    t = math.pi * math.sqrt(a ** 3 / mu)
    return abs(dv1), abs(dv2), t


def transfer_to_moon(home: BodyInfo, moon: BodyInfo, r_park: float
                     ) -> tuple[float, float, float]:
    """TLI-импульс, скорость на бесконечности у Луны и время перелёта."""
    r_m = moon.sma
    a = (r_park + r_m) / 2.0
    v_park = math.sqrt(home.mu / r_park)
    v_peri = math.sqrt(home.mu * (2 / r_park - 1 / a))
    v_apo = math.sqrt(home.mu * (2 / r_m - 1 / a))
    v_moon = math.sqrt(home.mu / r_m)
    t = math.pi * math.sqrt(a ** 3 / home.mu)
    return v_peri - v_park, abs(v_moon - v_apo), t


def capture_dv(moon: BodyInfo, v_inf: float, altitude: float) -> float:
    rp = moon.radius + altitude
    v_hyp = math.sqrt(v_inf ** 2 + 2 * moon.mu / rp)
    return v_hyp - math.sqrt(moon.mu / rp)


def landing_dv(moon: BodyInfo, from_altitude: float) -> float:
    """Сход с орбиты, гашение орбитальной скорости и зависание перед касанием."""
    v = moon.v_circ(from_altitude)
    return v * 1.12 + 50.0


def phase_angle_required(home: BodyInfo, moon: BodyInfo, r_park: float) -> float:
    """φ, при котором надо начинать перелёт: Луна должна прийти в апоцентр
    одновременно с аппаратом. Градусы."""
    _, _, t = transfer_to_moon(home, moon, r_park)
    return 180.0 - 360.0 * t / moon.period


# ==========================================================================
def budget(world: World, target: Target, vessel=None, gear: bool = False) -> Budget:
    """gear=True: assume a heat shield and parachutes (the designer adds them)."""
    h, m = world.home_body, world.moon_body
    lat = world.site.get("latitude", 0.0)
    h_leo = leo_altitude(h)
    r_leo = h.radius + h_leo
    b = Budget(target)

    asc = ascent_dv(h, h_leo, lat)
    b.legs.append(Maneuver("ascent", L(f"Ascent to LEO, H = {h_leo/1000:.0f} km", f"Выведение на НОО, H = {h_leo/1000:.0f} км"),
                           asc, h.name, atmospheric=True, twr_min=1.2,
                           note=L("planet rotation and atmospheric losses included", "с учётом вращения планеты и потерь в атмосфере")))
    t_asc = 300.0
    b.duration = t_asc + h.period_at(h_leo)

    if target.code == "LEO":
        return b

    if target.planet:
        from .planets import budget_legs
        heat = chutes = gear
        if vessel is not None:
            heat = any(p.resources_max.get("Ablator", 0) > 0 for p in vessel.parts)
            chutes = vessel.count("parachute") > 0
        legs, duration, _ = budget_legs(world, target, heat, chutes)
        b.legs += legs
        b.duration += duration
        return b

    if target.code == "HEO":
        dv1, dv2, t = hohmann(h, r_leo, h.radius + target.altitude)
        b.legs.append(Maneuver("raise", L(f"Raise Hα to {target.altitude/1000:,.0f} km", f"Подъём Hα до {target.altitude/1000:,.0f} км").replace(",", " "),
                               dv1, h.name, twr_min=0.3))
        b.legs.append(Maneuver("circ_high", L("High orbit circularization", "Скругление высокой орбиты"), dv2, h.name,
                               twr_min=0.2))
        b.duration += t + h.period_at(h_leo)
        return b

    tli, v_inf, t_tr = transfer_to_moon(h, m, r_leo)
    b.legs.append(Maneuver("tli", L(f"Transfer burn to {m.name} (TLI)", f"Перелётный импульс к {m.name} (TLI)"), tli, h.name,
                           twr_min=0.3))
    b.legs.append(Maneuver("mcc", L("Course corrections (reserve)", "Коррекции курса (резерв)"), 0.02 * tli + 20.0, h.name))
    alt = target.altitude if not target.landing else llo_altitude(m)
    loi = capture_dv(m, v_inf, alt)
    b.legs.append(Maneuver("loi", L(f"Capture at {m.name} (LOI), Hπ = {alt/1000:.0f} km", f"Торможение у {m.name} (LOI), Hπ = {alt/1000:.0f} км"),
                           loi, m.name, twr_min=0.25))
    synodic = 1.0 / (1.0 / h.period_at(h_leo) - 1.0 / m.period)
    b.duration += synodic / 2 + t_tr + m.period_at(alt)
    if target.landing:
        b.legs.append(Maneuver("land", L(f"Landing on {m.name}", f"Посадка на {m.name}"), landing_dv(m, alt),
                               m.name, twr_min=2.0,
                               note=L("deorbit, braking, touchdown", "сход с орбиты, гашение скорости, касание")))
        b.duration += 1200.0
    return b


# ==========================================================================
# Геометрия старта (работает только с живой игрой)
# ==========================================================================
def _v(a):
    return (float(a[0]), float(a[1]), float(a[2]))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _rotate(v, axis, angle):
    """Поворот вектора v вокруг оси axis (единичной) на angle, формула Родрига."""
    c, s = math.cos(angle), math.sin(angle)
    kxv = _cross(axis, v)
    kdv = _dot(axis, v)
    return tuple(v[i] * c + kxv[i] * s + axis[i] * kdv * (1 - c) for i in range(3))


def _axis_for(r, v):
    """Ось, поворот вокруг которой (через _rotate) ведёт r в сторону v.

    Системы отсчёта kRPC левосторонние, поэтому знак векторного
    произведения не угадывается, а проверяется."""
    axis = _norm(_cross(r, v))
    probe = _sub(_rotate(r, axis, 1e-3), r)
    return axis if _dot(probe, v) > 0 else _scale(axis, -1.0)


@dataclass
class LaunchGeometry:
    heading: float = 90.0            # азимут пуска с поправкой на вращение, °
    inclination: float = 0.0         # наклонение плоскости цели, °
    window_wait: float = 0.0         # ожидание окна пуска, с
    phase_at_orbit: float | None = None   # φ Луны в момент выхода на НОО, °
    phase_required: float | None = None
    tli_wait: float | None = None    # ожидание на НОО до TLI, с
    live: bool = False
    notes: list[str] = field(default_factory=list)


ASCENT_TIME = 300.0        # от отрыва до скругления, с
DOWNRANGE_DEG = 35.0       # на сколько градусов по дуге уходит ракета за выведение


def launch_geometry(world: World, target: Target) -> LaunchGeometry:
    g = LaunchGeometry()
    h, m = world.home_body, world.moon_body
    h_leo = leo_altitude(h)
    needs_moon = target.body == m.name and m.name != h.name
    if needs_moon:
        g.phase_required = phase_angle_required(h, m, h.radius + h_leo)
        g.inclination = math.degrees(m.inclination)

    conn = world.connection
    if conn is None:
        g.notes.append(L("game not connected: window and φ will be refined at launch", "игра не подключена: окно и φ будут уточнены на старте"))
        if needs_moon:
            synodic = 1.0 / (1.0 / h.period_at(h_leo) - 1.0 / m.period)
            g.tli_wait = synodic / 2
        return g
    try:
        sc = conn.space_center
        vessel = sc.active_vessel
        home = sc.bodies[h.name]
        frame = home.non_rotating_reference_frame
        r_site = _v(vessel.position(frame))
        v_site = _v(vessel.velocity(frame))
        ut = sc.ut
    except Exception as exc:
        g.notes.append(L(f"no launch data ({type(exc).__name__})", f"нет данных о старте ({type(exc).__name__})"))
        return g
    g.live = True
    omega = 2 * math.pi / h.rotation_period
    pole = _axis_for(r_site, v_site)

    if needs_moon:
        moon = sc.bodies[m.name]
        r_m = _v(moon.position(frame))
        v_m = _v(moon.velocity(frame))
        n_m = _cross(r_m, v_m)

        # --- Окно пуска: стартовая площадка проходит через плоскость орбиты Луны
        wait = 0.0
        lat = abs(world.site.get("latitude", 0.0))
        if g.inclination > max(0.5, lat):
            def off_plane(t):
                return _dot(_norm(n_m), _norm(_rotate(r_site, pole, omega * t)))
            step = h.rotation_period / 360.0
            prev = off_plane(0.0)
            t = 0.0
            while t < h.rotation_period:
                t2 = t + step
                cur = off_plane(t2)
                if prev == 0 or (prev < 0) != (cur < 0):
                    lo, hi = t, t2
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        if (off_plane(lo) < 0) != (off_plane(mid) < 0):
                            hi = mid
                        else:
                            lo = mid
                    wait = (lo + hi) / 2
                    break
                prev, t = cur, t2
            g.window_wait = wait
        elif g.inclination > 0.5:
            g.notes.append(L(f"target inclination {g.inclination:.1f}° is below launch latitude "
                             f"{lat:.1f}° — launching into the lowest possible inclination",
                             f"наклонение цели {g.inclination:.1f}° меньше широты старта "
                             f"{lat:.1f}° — пуск в минимально возможное наклонение"))

        # --- Азимут: направление орбитального движения в плоскости Луны
        r_launch = _rotate(r_site, pole, omega * wait)
        up = _norm(r_launch)
        east = _norm(_rotate(v_site, pole, omega * wait))
        # В системах тел kRPC ось y направлена на северный полюс.
        y_axis = (0.0, 1.0, 0.0)
        north = _norm(_sub(y_axis, _scale(up, _dot(y_axis, up))))
        if g.inclination > max(0.5, lat):
            u = _norm(_cross(n_m, r_launch))
        else:
            u = east
        beta = math.atan2(_dot(u, east), _dot(u, north))
        v_orb = h.v_circ(h_leo)
        v_rot = math.sqrt(_dot(v_site, v_site))
        g.heading = math.degrees(math.atan2(v_orb * math.sin(beta) - v_rot,
                                            v_orb * math.cos(beta))) % 360.0

        # --- Фазовый угол в момент выхода на НОО
        t_ins = wait + ASCENT_TIME
        r_ship = _rotate(r_site, pole, omega * t_ins)
        r_ship = _rotate(r_ship, pole, math.radians(DOWNRANGE_DEG))
        moon_rate = 2 * math.pi / m.period
        r_m_ins = _rotate(r_m, _axis_for(r_m, v_m), moon_rate * t_ins)
        s = _dot(_cross(r_ship, r_m_ins), _cross(r_ship, _rotate(r_ship, pole, 0.01)))
        c = _dot(_norm(r_ship), _norm(r_m_ins))
        sin_part = math.sqrt(max(0.0, 1 - c * c)) * (1 if s >= 0 else -1)
        phase = math.degrees(math.atan2(sin_part, c)) % 360.0
        g.phase_at_orbit = phase
        rate = 360.0 / h.period_at(h_leo) - 360.0 / m.period      # °/с
        wait_tli = ((phase - g.phase_required) % 360.0) / rate
        if wait_tli < 300.0:          # меньше 5 минут на проверку орбиты — следующий виток
            wait_tli += 360.0 / rate
        g.tli_wait = wait_tli
    else:
        lat = world.site.get("latitude", 0.0)
        v_orb = h.v_circ(h_leo)
        v_rot = math.sqrt(_dot(v_site, v_site))
        g.heading = math.degrees(math.atan2(v_orb - v_rot, 0.0)) % 360.0
    _ = ut
    return g
