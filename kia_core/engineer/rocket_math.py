"""Ракетная математика: Циолковский, TWR, ступени, орбитальная механика.

Модуль полностью автономен (не требует kRPC) — используется как при
проектировании в «уме», так и для проверки телеметрии в полёте.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import G0
from .parts_db import Body, Part, get_body


# ==========================================================================
# Базовые формулы
# ==========================================================================
def delta_v(isp: float, wet_mass: float, dry_mass: float) -> float:
    """Формула Циолковского: dV = Isp * g0 * ln(m0 / m1)."""
    if dry_mass <= 0 or wet_mass <= dry_mass:
        return 0.0
    return isp * G0 * math.log(wet_mass / dry_mass)


def burn_time(isp: float, thrust: float, wet_mass: float, dry_mass: float) -> float:
    """Время работы ступени на полной тяге до сухой массы."""
    if thrust <= 0 or isp <= 0:
        return 0.0
    mass_flow = thrust * 1000.0 / (isp * G0)   # кг/с
    return (wet_mass - dry_mass) * 1000.0 / mass_flow


def mass_after_burn(wet_mass: float, dv: float, isp: float) -> float:
    """Масса после манёвра заданного dV."""
    return wet_mass * math.exp(-dv / (isp * G0))


def fuel_for_burn(wet_mass: float, dv: float, isp: float) -> float:
    """Расход массы (т) на манёвр заданного dV."""
    return wet_mass - mass_after_burn(wet_mass, dv, isp)


def twr(thrust_kn: float, mass_t: float, gravity: float = None) -> float:
    """Отношение тяги к весу. gravity по умолчанию — поверхность Кербина."""
    if mass_t <= 0:
        return 0.0
    g = gravity if gravity is not None else get_body("Kerbin").surface_gravity
    return (thrust_kn * 1000.0) / (mass_t * 1000.0 * g)


def required_thrust(mass_t: float, target_twr: float, gravity: float = None) -> float:
    g = gravity if gravity is not None else get_body("Kerbin").surface_gravity
    return target_twr * mass_t * g


def combined_isp(engines: list[tuple[float, float]]) -> float:
    """Эффективный Isp связки: sum(F) / sum(F/Isp). engines = [(thrust, isp), ...]"""
    total_thrust = sum(t for t, _ in engines)
    denom = sum(t / i for t, i in engines if i > 0)
    if denom <= 0:
        return 0.0
    return total_thrust / denom


def atmospheric_isp(part: Part, pressure_atm: float) -> float:
    """Линейная интерполяция Isp по давлению (достаточная аппроксимация)."""
    p = max(0.0, min(1.0, pressure_atm))
    return part.isp_vac + (part.isp_asl - part.isp_vac) * p


def atmospheric_thrust(part: Part, pressure_atm: float) -> float:
    p = max(0.0, min(1.0, pressure_atm))
    return part.thrust_vac + (part.thrust_asl - part.thrust_vac) * p


def atmosphere_pressure(body: Body, altitude: float) -> float:
    """Экспоненциальная модель атмосферы (атм). Для Кербина H ≈ 5600 м."""
    if body.atmosphere_height <= 0 or altitude >= body.atmosphere_height:
        return 0.0
    scale_height = body.atmosphere_height / 12.5
    return math.exp(-altitude / scale_height)


# ==========================================================================
# Ступени
# ==========================================================================
@dataclass
class StageSpec:
    """Расчётное описание одной ступени ракеты."""
    index: int
    engine: Part
    engine_count: int
    tanks: list[Part] = field(default_factory=list)
    decoupler: Part | None = None
    extra_mass: float = 0.0          # полезная нагрузка/наука на этой ступени
    payload_mass: float = 0.0        # масса всего, что находится выше

    # ---- массы ----
    @property
    def engines_mass(self) -> float:
        return self.engine.dry_mass * self.engine_count

    @property
    def tanks_dry(self) -> float:
        return sum(t.dry_mass for t in self.tanks)

    @property
    def fuel_mass(self) -> float:
        return sum(t.fuel_mass for t in self.tanks)

    @property
    def decoupler_mass(self) -> float:
        return self.decoupler.dry_mass if self.decoupler else 0.0

    @property
    def stage_dry_mass(self) -> float:
        return (self.engines_mass + self.tanks_dry + self.decoupler_mass
                + self.extra_mass)

    @property
    def wet_mass(self) -> float:
        return self.payload_mass + self.stage_dry_mass + self.fuel_mass

    @property
    def dry_mass(self) -> float:
        return self.payload_mass + self.stage_dry_mass

    # ---- тяга ----
    def thrust(self, pressure_atm: float = 0.0) -> float:
        return atmospheric_thrust(self.engine, pressure_atm) * self.engine_count

    def isp(self, pressure_atm: float = 0.0) -> float:
        return atmospheric_isp(self.engine, pressure_atm)

    # ---- характеристики ----
    def delta_v(self, pressure_atm: float = 0.0) -> float:
        return delta_v(self.isp(pressure_atm), self.wet_mass, self.dry_mass)

    def twr(self, body_name: str = "Kerbin", pressure_atm: float = 0.0) -> float:
        body = get_body(body_name)
        return twr(self.thrust(pressure_atm), self.wet_mass, body.surface_gravity)

    def burn_time(self, pressure_atm: float = 0.0) -> float:
        return burn_time(self.isp(pressure_atm), self.thrust(pressure_atm),
                         self.wet_mass, self.dry_mass)

    def summary(self) -> dict:
        return {
            "index": self.index,
            "engine": self.engine.title,
            "engine_count": self.engine_count,
            "tanks": [t.title for t in self.tanks],
            "wet_mass_t": round(self.wet_mass, 3),
            "dry_mass_t": round(self.dry_mass, 3),
            "dv_vac": round(self.delta_v(0.0), 1),
            "dv_asl": round(self.delta_v(1.0), 1),
            "twr_asl": round(self.twr("Kerbin", 1.0), 3),
            "twr_vac": round(self.twr("Kerbin", 0.0), 3),
            "burn_time_s": round(self.burn_time(0.0), 1),
        }


def total_delta_v(stages: list[StageSpec], pressure_profile: list[float] | None = None) -> float:
    """Суммарный dV ракеты. pressure_profile — давление для каждой ступени."""
    if pressure_profile is None:
        pressure_profile = [1.0 if s.index == 0 else 0.0 for s in stages]
    return sum(s.delta_v(p) for s, p in zip(stages, pressure_profile))


def link_stage_masses(stages: list[StageSpec], payload_mass: float) -> None:
    """Проставляет payload_mass снизу вверх: stages[0] — нижняя ступень."""
    accumulated = payload_mass
    for stage in reversed(stages):
        stage.payload_mass = accumulated
        accumulated += stage.stage_dry_mass + stage.fuel_mass


# ==========================================================================
# Орбитальная механика
# ==========================================================================
def orbital_speed(body: Body, radius: float, semi_major_axis: float) -> float:
    """Vis-viva: v = sqrt(mu * (2/r - 1/a))."""
    return math.sqrt(max(0.0, body.mu * (2.0 / radius - 1.0 / semi_major_axis)))


def circular_speed(body: Body, altitude: float) -> float:
    return math.sqrt(body.mu / (body.radius + altitude))


def orbital_period(body: Body, semi_major_axis: float) -> float:
    return 2 * math.pi * math.sqrt(semi_major_axis ** 3 / body.mu)


def hohmann_transfer(body: Body, r1: float, r2: float) -> tuple[float, float, float]:
    """Гомановский переход между круговыми орбитами радиусов r1 -> r2.

    Возвращает (dv1, dv2, время перелёта).
    """
    a_t = (r1 + r2) / 2.0
    v1 = math.sqrt(body.mu / r1)
    v2 = math.sqrt(body.mu / r2)
    v_peri = orbital_speed(body, r1, a_t)
    v_apo = orbital_speed(body, r2, a_t)
    tof = math.pi * math.sqrt(a_t ** 3 / body.mu)
    return v_peri - v1, v2 - v_apo, tof


def transfer_phase_angle(body: Body, r1: float, r2: float) -> float:
    """Требуемый фазовый угол (радианы) цели относительно корабля для Гомана."""
    a_t = (r1 + r2) / 2.0
    tof = math.pi * math.sqrt(a_t ** 3 / body.mu)
    target_omega = math.sqrt(body.mu / r2 ** 3)
    return math.pi - target_omega * tof


def circularization_dv(body: Body, apoapsis_alt: float, periapsis_alt: float) -> float:
    """dV для скругления орбиты в апоапсисе."""
    ra = body.radius + apoapsis_alt
    rp = body.radius + periapsis_alt
    a = (ra + rp) / 2.0
    v_apo = orbital_speed(body, ra, a)
    v_circ = math.sqrt(body.mu / ra)
    return v_circ - v_apo


def sphere_of_influence_capture_dv(target: Body, arrival_speed: float,
                                   periapsis_alt: float) -> float:
    """dV захвата на круговую орбиту у цели при заданной скорости входа в SOI."""
    rp = target.radius + periapsis_alt
    v_inf_sq = max(0.0, arrival_speed ** 2)
    v_peri = math.sqrt(v_inf_sq + 2 * target.mu / rp)
    v_circ = math.sqrt(target.mu / rp)
    return v_peri - v_circ


# --------------------------------------------------------------------------
# Сложные манёвры (ступень математика «Матан+++»)
# --------------------------------------------------------------------------
def bielliptic_dv(mu: float, r1: float, r2: float, rb: float) -> float:
    """Двухэллиптический переход r1 → rb → r2: сумма трёх импульсов.

    При r2/r1 > ~11.9 дешевле Гомана: большой разгон к далёкому апоцентру,
    там дешёвый подъём перицентра, затем торможение до круговой.
    """
    a1 = (r1 + rb) / 2.0
    a2 = (r2 + rb) / 2.0
    dv1 = math.sqrt(mu * (2 / r1 - 1 / a1)) - math.sqrt(mu / r1)
    dv2 = math.sqrt(mu * (2 / rb - 1 / a2)) - math.sqrt(mu * (2 / rb - 1 / a1))
    dv3 = math.sqrt(mu * (2 / r2 - 1 / a2)) - math.sqrt(mu / r2)
    return abs(dv1) + abs(dv2) + abs(dv3)


def two_stage_dv(isp1: float, wet1: float, dry1: float,
                 isp2: float, wet2: float, dry2: float, payload: float) -> float:
    """Полная ΔV двухступенчатой ракеты с полезной нагрузкой (массы в т).

    Первая ступень везёт вторую со всем топливом и нагрузку; вторая —
    только нагрузку. Сумма двух Циолковских со «стопкой» масс.
    """
    upper = wet2 + payload
    dv1 = isp1 * G0 * math.log((wet1 + upper) / (dry1 + upper))
    dv2 = isp2 * G0 * math.log((wet2 + payload) / (dry2 + payload))
    return dv1 + dv2


def ejection_dv(mu: float, r_park: float, v_inf: float) -> float:
    """Импульс ухода с круговой опорной орбиты с избытком v∞ (эффект Оберта)."""
    return math.sqrt(v_inf * v_inf + 2 * mu / r_park) - math.sqrt(mu / r_park)


def combined_plane_change(v1: float, v2: float, delta_i: float) -> float:
    """Один импульс: скорость v1 → v2 с поворотом плоскости на delta_i (рад).

    Совмещать скругление со сменой плоскости дешевле, чем делать их по
    отдельности, — это правило косинусов для векторов скорости.
    """
    return math.sqrt(max(0.0, v1 * v1 + v2 * v2 - 2 * v1 * v2 * math.cos(delta_i)))


def suicide_burn(v_down: float, thrust_kn: float, mass_t: float,
                 g: float) -> tuple[float, float]:
    """Посадка «в последний момент»: (высота начала торможения м, время с).

    Двигатель на полную, вертикальное падение: замедление a = F/m − g.
    """
    a = thrust_kn / mass_t - g
    if a <= 0:
        return math.inf, math.inf
    return v_down * v_down / (2 * a), v_down / a


def soi_radius(sma: float, mu_body: float, mu_parent: float) -> float:
    """Радиус сферы действия (Лаплас): a · (μ/M)^(2/5)."""
    return sma * (mu_body / mu_parent) ** 0.4


def time_to_phase_angle(current_angle: float, target_angle: float,
                        omega_ship: float, omega_target: float) -> float:
    """Сколько секунд ждать, пока фазовый угол станет целевым."""
    rel_omega = omega_target - omega_ship
    if abs(rel_omega) < 1e-12:
        return float("inf")
    delta = (target_angle - current_angle) % (2 * math.pi)
    if rel_omega < 0:
        delta = delta - 2 * math.pi
    return abs(delta / rel_omega)


# ==========================================================================
# Задача Ламберта: перелёт между двумя точками за заданное время
# ==========================================================================
def _stumpff_c(z: float) -> float:
    if z > 1e-6:
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-6:
        return (math.cosh(math.sqrt(-z)) - 1) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def _stumpff_s(z: float) -> float:
    if z > 1e-6:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s ** 3)
    if z < -1e-6:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s ** 3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def lambert(r1: tuple, r2: tuple, tof: float, mu: float,
            prograde: bool = True, tolerance: float = 1e-8,
            max_iterations: int = 80) -> tuple[tuple, tuple] | None:
    """Решение задачи Ламберта методом универсальных переменных (Bate–Mueller–White).

    По двум радиус-векторам и времени перелёта возвращает (v1, v2) —
    скорости в начальной и конечной точках. None, если решение не найдено.

    Это ядро планирования перелётов: система сама считает, с какой
    скоростью нужно уйти с опорной орбиты, чтобы попасть в точку встречи.
    """
    r1n = math.sqrt(sum(c * c for c in r1))
    r2n = math.sqrt(sum(c * c for c in r2))
    if r1n <= 0 or r2n <= 0 or tof <= 0 or mu <= 0:
        return None

    cross_z = r1[0] * r2[2] - r1[2] * r2[0]      # компонента «вверх» в системе KSP
    dot = sum(a * b for a, b in zip(r1, r2))
    cos_dnu = clamp(dot / (r1n * r2n), -1.0, 1.0)
    dnu = math.acos(cos_dnu)
    if prograde and cross_z > 0:
        dnu = 2 * math.pi - dnu
    elif not prograde and cross_z < 0:
        dnu = 2 * math.pi - dnu

    sin_dnu = math.sin(dnu)
    if abs(sin_dnu) < 1e-12:
        return None
    A = sin_dnu * math.sqrt(r1n * r2n / (1 - cos_dnu)) if (1 - cos_dnu) > 1e-12 else 0.0
    if abs(A) < 1e-12:
        return None

    z, lo, hi = 0.0, -4 * math.pi ** 2, 4 * math.pi ** 2
    y = r1n + r2n
    for _ in range(max_iterations):
        c, s = _stumpff_c(z), _stumpff_s(z)
        y = r1n + r2n + A * (z * s - 1.0) / math.sqrt(c) if c > 0 else r1n + r2n
        if A > 0 and y < 0:
            lo = z
            z = (z + hi) / 2.0
            continue
        if c <= 0 or y <= 0:
            lo = z
            z = (z + hi) / 2.0
            continue
        x = math.sqrt(y / c)
        t = (x ** 3 * s + A * math.sqrt(y)) / math.sqrt(mu)
        if abs(t - tof) < tolerance * max(1.0, tof):
            break
        if t < tof:
            lo = z
        else:
            hi = z
        z = (lo + hi) / 2.0
    else:
        return None

    c = _stumpff_c(z)
    if c <= 0 or y <= 0:
        return None
    f = 1 - y / r1n
    g = A * math.sqrt(y / mu)
    g_dot = 1 - y / r2n
    if abs(g) < 1e-12:
        return None
    v1 = tuple((r2[i] - f * r1[i]) / g for i in range(3))
    v2 = tuple((g_dot * r2[i] - r1[i]) / g for i in range(3))
    return v1, v2


def lambert_injection_dv(r1: tuple, v1_current: tuple, r2: tuple, tof: float,
                         mu: float) -> tuple[float, tuple] | None:
    """Импульс, переводящий корабль на траекторию встречи (по Ламберту).

    Возвращает (модуль dV, вектор dV) или None, если решения нет.
    """
    solution = lambert(r1, r2, tof, mu)
    if solution is None:
        return None
    v_needed, _ = solution
    dv_vec = tuple(v_needed[i] - v1_current[i] for i in range(3))
    return math.sqrt(sum(c * c for c in dv_vec)), dv_vec


def propagate_body_position(radius: float, angle0: float, period: float,
                            dt: float) -> tuple:
    """Положение тела на круговой орбите через dt секунд (плоская задача)."""
    omega = 2 * math.pi / period if period else 0.0
    angle = angle0 + omega * dt
    return (radius * math.cos(angle), 0.0, radius * math.sin(angle))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    """Приводит угол к [0, 2*pi)."""
    return angle % (2 * math.pi)


def angle_between(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return math.acos(clamp(dot / (n1 * n2), -1.0, 1.0))
