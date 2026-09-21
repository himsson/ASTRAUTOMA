"""Пространства состояний и действий нейросети KIA.

ВАЖНО о «знании физики». Сеть не получает ни одной производной величины,
посчитанной по формулам: ни ΔV, ни TWR, ни фазовых углов. На вход идут
сырые показания kRPC. Единственное, что делается с числами, — деление на
масштабный коэффициент, чтобы они попали в диапазон примерно [-3, 3]:
без этого градиентный спуск расходится на входах порядка 10^6 (высота в
метрах) вперемешку с числами порядка 0.1. Масштабы — это про численную
устойчивость, а не про физику: сеть по-прежнему не знает, что такое
апоапсис и почему он важен.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------
# ПОЛЁТ: наблюдение
# --------------------------------------------------------------------------
# (имя, масштаб) — значение делится на масштаб
FLIGHT_FEATURES: list[tuple[str, float]] = [
    ("altitude", 100_000.0),
    ("surface_altitude", 100_000.0),
    ("apoapsis", 100_000.0),
    ("periapsis", 100_000.0),
    ("time_to_apoapsis", 300.0),
    ("eccentricity", 1.0),
    ("inclination", 1.0),
    ("speed", 2_500.0),
    ("vertical_speed", 500.0),
    ("horizontal_speed", 2_500.0),
    ("pitch_sin", 1.0),
    ("pitch_cos", 1.0),
    ("heading_sin", 1.0),
    ("heading_cos", 1.0),
    ("roll_sin", 1.0),
    ("roll_cos", 1.0),
    ("prograde_pitch_sin", 1.0),
    ("prograde_pitch_cos", 1.0),
    ("angle_of_attack", 90.0),
    ("dynamic_pressure", 20_000.0),
    ("static_pressure", 101_325.0),
    ("atmosphere_density", 1.25),
    ("g_force", 5.0),
    ("mass", 20_000.0),
    ("dry_mass", 10_000.0),
    ("thrust", 500_000.0),
    ("available_thrust", 500_000.0),
    ("specific_impulse", 350.0),
    ("throttle", 1.0),
    ("stage", 8.0),
    ("part_count", 40.0),
    ("liquid_fuel_fraction", 1.0),
    ("oxidizer_fraction", 1.0),
    ("electric_fraction", 1.0),
    ("mission_time", 600.0),
    ("body_gravity", 10.0),
    ("body_radius", 1_000_000.0),
    ("has_atmosphere", 1.0),
    ("goal_altitude", 100_000.0),
    ("goal_progress", 1.0),
]

FLIGHT_OBS_DIM = len(FLIGHT_FEATURES)

# Потолок апоапсиса и периапсиса на ВХОДЕ сети, в целевых высотах.
#
# На гиперболической траектории апоапсиса не существует, и симулятор
# отдаёт 1e9, а периапсис уходит на минус миллионы. После деления на
# масштаб 100 000 это даёт признаки 10 000 и −61 при обычном рабочем
# диапазоне ±10. Бегущий нормализатор такие выбросы не отбрасывает: он
# считает по ним дисперсию, она взлетает, и ВСЕ высотные признаки
# схлопываются к нулю — сеть перестаёт различать 5 км и 50 км. Хуже
# того, испорченная статистика лежит в чекпоинте и переживает перезапуск.
#
# Ограничение здесь, в кодировщике, а не в тренажёре: живая KSP на
# гиперболе отдаёт такой же мусор, а вектор наблюдения обязан совпадать.
# Эксцентриситет по той же причине давно режется по 3.0.
OBSERVED_APOAPSIS_LIMIT = 10.0        # целевых высот, дальше сети всё равно
OBSERVED_TIME_LIMIT = 1_800.0         # с, «время до апоапсиса» на гиперболе

# Общий предохранитель: ни один признак не имеет права уйти дальше этого
# по модулю. Масштабы подобраны так, чтобы рабочие значения лежали в
# пределах ±3; всё, что за ±50, — это не информация, а сбой источника
# (гипербола, деление на почти ноль, мусор из телеметрии при обрыве
# связи). Отдельные потолки выше объясняют СМЫСЛ конкретных величин, а
# этот — последняя линия обороны для тех, о которых ещё не подумали.
# Дёшево, а стоило проекту всего обучения: см. OBSERVED_APOAPSIS_LIMIT.
OBSERVED_FEATURE_LIMIT = 50.0

# --------------------------------------------------------------------------
# ПОЛЁТ: действия
# --------------------------------------------------------------------------
# Непрерывные выходы сети лежат в [-1, 1] и переводятся в команды KSP:
#   throttle: [-1,1] -> [0,1]
#   pitch/yaw/roll: напрямую как команды осей управления
CONTINUOUS_ACTIONS = ["throttle", "pitch", "yaw", "roll"]
CONTINUOUS_DIM = len(CONTINUOUS_ACTIONS)

# Дискретные головы: [не отделять / отделить ступень]
DISCRETE_ACTIONS = [("stage", 2)]


@dataclass
class FlightAction:
    """Расшифрованное действие пилота."""
    throttle: float
    pitch: float
    yaw: float
    roll: float
    stage: bool

    @classmethod
    def decode(cls, continuous, discrete) -> "FlightAction":
        raw = [float(v) for v in continuous]
        clipped = [max(-1.0, min(1.0, v)) for v in raw]
        return cls(
            throttle=(clipped[0] + 1.0) / 2.0,
            pitch=clipped[1],
            yaw=clipped[2],
            roll=clipped[3],
            stage=bool(int(discrete[0])) if len(discrete) else False,
        )

    def describe(self) -> str:
        return (f"тяга {self.throttle:.2f} | тангаж {self.pitch:+.2f} | "
                f"рыскание {self.yaw:+.2f} | крен {self.roll:+.2f}"
                + (" | ОТДЕЛЕНИЕ" if self.stage else ""))


# --------------------------------------------------------------------------
def encode_flight_state(snap, goal_altitude: float, body_gravity: float,
                        body_radius: float, has_atmosphere: bool,
                        prograde_pitch: float = 0.0,
                        angle_of_attack: float = 0.0) -> list[float]:
    """Сырая телеметрия -> вектор признаков фиксированной длины.

    Углы подаются парой (sin, cos): иначе сеть видит разрыв на 359°/0°.
    """
    lf = snap.resources.get("LiquidFuel", {})
    ox = snap.resources.get("Oxidizer", {})
    ec = snap.resources.get("ElectricCharge", {})

    def fraction(res: dict) -> float:
        top = res.get("max", 0.0) or 0.0
        return (res.get("amount", 0.0) / top) if top > 0 else 0.0

    # Гиперболический «апоапсис» 1e9 и периапсис в минус миллионы —
    # не информация, а выброс, который портит нормализатор (см.
    # OBSERVED_APOAPSIS_LIMIT). Режем оба до разумных пределов.
    ceiling = OBSERVED_APOAPSIS_LIMIT * (goal_altitude or 100_000.0)
    apoapsis = max(-ceiling, min(ceiling, float(snap.apoapsis or 0.0)))
    periapsis = max(-ceiling, min(ceiling, float(snap.periapsis or 0.0)))
    # «Время до апоапсиса» считается из того же апоапсиса и на гиперболе
    # улетает вслед за ним. В живых весах этот вход тоже оказался мёртв.
    time_to_apoapsis = max(0.0, min(OBSERVED_TIME_LIMIT,
                                    float(snap.time_to_apoapsis or 0.0)))

    pitch_rad = math.radians(snap.pitch or 0.0)
    heading_rad = math.radians(snap.heading or 0.0)
    roll_rad = math.radians(snap.roll or 0.0)
    prograde_rad = math.radians(prograde_pitch or 0.0)

    values = {
        "altitude": snap.altitude,
        "surface_altitude": snap.surface_altitude,
        "apoapsis": apoapsis,
        "periapsis": periapsis,
        "time_to_apoapsis": time_to_apoapsis,
        "eccentricity": snap.eccentricity,
        "inclination": snap.inclination,
        "speed": snap.speed,
        "vertical_speed": snap.vertical_speed,
        "horizontal_speed": snap.horizontal_speed,
        "pitch_sin": math.sin(pitch_rad),
        "pitch_cos": math.cos(pitch_rad),
        "heading_sin": math.sin(heading_rad),
        "heading_cos": math.cos(heading_rad),
        "roll_sin": math.sin(roll_rad),
        "roll_cos": math.cos(roll_rad),
        "prograde_pitch_sin": math.sin(prograde_rad),
        "prograde_pitch_cos": math.cos(prograde_rad),
        "angle_of_attack": angle_of_attack,
        "dynamic_pressure": snap.dynamic_pressure,
        "static_pressure": snap.static_pressure,
        "atmosphere_density": snap.atmosphere_density,
        "g_force": snap.g_force,
        "mass": snap.mass * 1000.0,
        "dry_mass": snap.dry_mass * 1000.0,
        "thrust": snap.thrust * 1000.0,
        "available_thrust": snap.available_thrust * 1000.0,
        "specific_impulse": snap.specific_impulse,
        "throttle": snap.throttle,
        "stage": float(snap.stage),
        "part_count": float(snap.part_count),
        "liquid_fuel_fraction": fraction(lf),
        "oxidizer_fraction": fraction(ox),
        "electric_fraction": fraction(ec),
        "mission_time": snap.mission_time,
        "body_gravity": body_gravity,
        "body_radius": body_radius,
        "has_atmosphere": 1.0 if has_atmosphere else 0.0,
        "goal_altitude": goal_altitude,
        "goal_progress": (apoapsis / goal_altitude) if goal_altitude else 0.0,
    }

    vector = []
    for name, scale in FLIGHT_FEATURES:
        raw = float(values.get(name, 0.0) or 0.0)
        if not math.isfinite(raw):
            raw = 0.0
        value = raw / scale if scale else raw
        vector.append(max(-OBSERVED_FEATURE_LIMIT,
                          min(OBSERVED_FEATURE_LIMIT, value)))
    return vector


# --------------------------------------------------------------------------
# СБОРКА: наблюдение и действия
# --------------------------------------------------------------------------
# Сеть-конструктор выбирает деталь за деталью. Действие — индекс детали
# в подготовленном срезе каталога либо специальный токен «закончить».
BUILDER_SLOT_LIMIT = 12          # максимум деталей в одной сборке

BUILDER_FEATURES: list[tuple[str, float]] = [
    ("slot_index", float(BUILDER_SLOT_LIMIT)),
    ("total_mass", 20_000.0),
    ("dry_mass", 10_000.0),
    ("fuel_units", 4_000.0),
    ("thrust_sum", 500_000.0),
    ("isp_mean", 350.0),
    ("part_count", float(BUILDER_SLOT_LIMIT)),
    ("has_command", 1.0),
    ("has_engine", 1.0),
    ("has_tank", 1.0),
    ("has_decoupler", 1.0),
    ("has_power", 1.0),
    ("last_is_engine", 1.0),
    ("last_is_tank", 1.0),
    ("last_diameter", 3.75),
    ("goal_altitude", 100_000.0),
    ("goal_transfer", 1.0),
    ("goal_landing", 1.0),
]

BUILDER_OBS_DIM = len(BUILDER_FEATURES)


@dataclass
class BuilderState:
    """Состояние частично собранной ракеты (только наблюдаемые числа)."""
    slot_index: int = 0
    total_mass: float = 0.0
    dry_mass: float = 0.0
    fuel_units: float = 0.0
    thrust_sum: float = 0.0
    isp_mean: float = 0.0
    part_count: int = 0
    has_command: bool = False
    has_engine: bool = False
    has_tank: bool = False
    has_decoupler: bool = False
    has_power: bool = False
    last_is_engine: bool = False
    last_is_tank: bool = False
    last_diameter: float = 0.0

    def encode(self, goal_altitude: float, goal_transfer: bool,
               goal_landing: bool) -> list[float]:
        values = {
            "slot_index": float(self.slot_index),
            "total_mass": self.total_mass,
            "dry_mass": self.dry_mass,
            "fuel_units": self.fuel_units,
            "thrust_sum": self.thrust_sum,
            "isp_mean": self.isp_mean,
            "part_count": float(self.part_count),
            "has_command": float(self.has_command),
            "has_engine": float(self.has_engine),
            "has_tank": float(self.has_tank),
            "has_decoupler": float(self.has_decoupler),
            "has_power": float(self.has_power),
            "last_is_engine": float(self.last_is_engine),
            "last_is_tank": float(self.last_is_tank),
            "last_diameter": self.last_diameter,
            "goal_altitude": goal_altitude,
            "goal_transfer": float(goal_transfer),
            "goal_landing": float(goal_landing),
        }
        return [float(values.get(name, 0.0)) / (scale or 1.0)
                for name, scale in BUILDER_FEATURES]
