"""Справочная база деталей KSP 1.12 (сток) и небесных тел.

Массы в тоннах, тяга в кН, Isp в секундах, длины в метрах.
Данные используются Инженером для расчётов ДО подключения к игре,
поэтому они захардкожены, а не читаются из ConfigNode'ов KSP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PartCategory(str, Enum):
    COMMAND = "command"
    TANK = "tank"
    ENGINE = "engine"
    DECOUPLER = "decoupler"
    SCIENCE = "science"
    UTILITY = "utility"
    PARACHUTE = "parachute"


@dataclass(frozen=True)
class Part:
    """Обобщённое описание детали."""
    part_id: str                  # внутреннее имя KSP (для .craft файла)
    title: str
    category: PartCategory
    dry_mass: float               # т
    cost: float = 0.0
    size: float = 1.25            # диаметр стыковочного узла, м
    # --- топливо ---
    liquid_fuel: float = 0.0      # единиц
    oxidizer: float = 0.0
    # --- двигатель ---
    thrust_vac: float = 0.0       # кН
    thrust_asl: float = 0.0       # кН
    isp_vac: float = 0.0          # с
    isp_asl: float = 0.0          # с
    gimbal: float = 0.0           # градусов
    # --- прочее ---
    electric_charge: float = 0.0
    crash_tolerance: float = 6.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fuel_mass(self) -> float:
        """Масса топлива: LF = 5 кг/ед., Ox = 5 кг/ед."""
        return (self.liquid_fuel + self.oxidizer) * 0.005

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.fuel_mass


# --------------------------------------------------------------------------
# Командные модули
# --------------------------------------------------------------------------
COMMAND_PARTS: dict[str, Part] = {
    "mk1pod_v2": Part("mk1pod_v2", "Mk1 Command Pod", PartCategory.COMMAND,
                      dry_mass=0.84, cost=600, electric_charge=50,
                      liquid_fuel=0.0, crash_tolerance=14,
                      tags=("crewed", "1.25m")),
    "probeCoreOcto2": Part("probeCoreOcto2", "Probodobodyne OKTO2", PartCategory.COMMAND,
                           dry_mass=0.04, cost=1480, size=0.625, electric_charge=5,
                           tags=("probe",)),
    "probeStackLarge": Part("probeStackLarge", "RC-L01 Remote Guidance", PartCategory.COMMAND,
                            dry_mass=0.5, cost=3400, size=2.5, electric_charge=270,
                            tags=("probe",)),
}

# --------------------------------------------------------------------------
# Топливные баки (LF+Ox, соотношение 9:11)
# --------------------------------------------------------------------------
TANK_PARTS: dict[str, Part] = {
    "fuelTankSmall": Part("fuelTankSmall", "FL-T100", PartCategory.TANK,
                          dry_mass=0.0625, cost=150, liquid_fuel=45, oxidizer=55),
    "fuelTank": Part("fuelTank", "FL-T400", PartCategory.TANK,
                     dry_mass=0.25, cost=500, liquid_fuel=180, oxidizer=220),
    "fuelTank_long": Part("fuelTank_long", "FL-T800", PartCategory.TANK,
                          dry_mass=0.5, cost=800, liquid_fuel=360, oxidizer=440),
    "Rockomax16": Part("Rockomax16", "Rockomax X200-8", PartCategory.TANK,
                       dry_mass=0.5, cost=800, size=2.5, liquid_fuel=360, oxidizer=440),
    "Rockomax32": Part("Rockomax32", "Rockomax X200-16", PartCategory.TANK,
                       dry_mass=1.0, cost=1550, size=2.5, liquid_fuel=720, oxidizer=880),
    "Rockomax64": Part("Rockomax64", "Rockomax Jumbo-64", PartCategory.TANK,
                       dry_mass=4.0, cost=5750, size=2.5, liquid_fuel=2880, oxidizer=3520),
}

# --------------------------------------------------------------------------
# Двигатели
# --------------------------------------------------------------------------
ENGINE_PARTS: dict[str, Part] = {
    "liquidEngine2": Part("liquidEngine2", "LV-T45 Swivel", PartCategory.ENGINE,
                          dry_mass=1.5, cost=1200,
                          thrust_vac=215.0, thrust_asl=167.97,
                          isp_vac=320, isp_asl=250, gimbal=3.0),
    "liquidEngine": Part("liquidEngine", "LV-T30 Reliant", PartCategory.ENGINE,
                         dry_mass=1.25, cost=1100,
                         thrust_vac=240.0, thrust_asl=205.16,
                         isp_vac=310, isp_asl=265, gimbal=0.0),
    "liquidEngine3_v2": Part("liquidEngine3_v2", "LV-909 Terrier", PartCategory.ENGINE,
                             dry_mass=0.5, cost=390,
                             thrust_vac=60.0, thrust_asl=14.78,
                             isp_vac=345, isp_asl=85, gimbal=4.0),
    "liquidEngine2-2": Part("liquidEngine2-2", "Rockomax Skipper", PartCategory.ENGINE,
                            dry_mass=3.0, cost=5300, size=2.5,
                            thrust_vac=650.0, thrust_asl=568.75,
                            isp_vac=320, isp_asl=280, gimbal=2.0),
    "liquidEngine1-2": Part("liquidEngine1-2", "Rockomax Mainsail", PartCategory.ENGINE,
                            dry_mass=6.0, cost=13000, size=2.5,
                            thrust_vac=1500.0, thrust_asl=1379.03,
                            isp_vac=310, isp_asl=285, gimbal=2.0),
}

# --------------------------------------------------------------------------
# Разделители, наука, утилиты
# --------------------------------------------------------------------------
STRUCTURAL_PARTS: dict[str, Part] = {
    "stackDecoupler": Part("stackDecoupler", "TD-12 Decoupler", PartCategory.DECOUPLER,
                           dry_mass=0.05, cost=400),
    "decoupler1-2": Part("decoupler1-2", "TD-25 Decoupler", PartCategory.DECOUPLER,
                         dry_mass=0.16, cost=550, size=2.5),
    "parachuteSingle": Part("parachuteSingle", "Mk16 Parachute", PartCategory.PARACHUTE,
                            dry_mass=0.1, cost=422),
}

SCIENCE_PARTS: dict[str, Part] = {
    "sensorThermometer": Part("sensorThermometer", "2HOT Thermometer", PartCategory.SCIENCE,
                              dry_mass=0.005, cost=900, size=0.0),
    "GooExperiment": Part("GooExperiment", "Mystery Goo Containment", PartCategory.SCIENCE,
                          dry_mass=0.1, cost=800, size=0.0),
    "sensorBarometer": Part("sensorBarometer", "PresMat Barometer", PartCategory.SCIENCE,
                            dry_mass=0.005, cost=880, size=0.0),
}

UTILITY_PARTS: dict[str, Part] = {
    "batteryPack": Part("batteryPack", "Z-100 Battery", PartCategory.UTILITY,
                        dry_mass=0.005, cost=80, electric_charge=100, size=0.0),
    "solarPanels5": Part("solarPanels5", "OX-STAT Photovoltaic", PartCategory.UTILITY,
                         dry_mass=0.005, cost=75, size=0.0),
    "longAntenna": Part("longAntenna", "Communotron 16", PartCategory.UTILITY,
                        dry_mass=0.005, cost=300, size=0.0),
}

ALL_PARTS: dict[str, Part] = {
    **COMMAND_PARTS, **TANK_PARTS, **ENGINE_PARTS,
    **STRUCTURAL_PARTS, **SCIENCE_PARTS, **UTILITY_PARTS,
}


def get_part(part_id: str) -> Part:
    if part_id not in ALL_PARTS:
        raise KeyError(f"Деталь '{part_id}' отсутствует в базе KIA")
    return ALL_PARTS[part_id]


def parts_by_category(category: PartCategory) -> list[Part]:
    return [p for p in ALL_PARTS.values() if p.category is category]


def engines_for_size(size: float) -> list[Part]:
    return sorted((p for p in ENGINE_PARTS.values() if abs(p.size - size) < 1e-6),
                  key=lambda p: p.thrust_vac)


def tanks_for_size(size: float) -> list[Part]:
    return sorted((p for p in TANK_PARTS.values() if abs(p.size - size) < 1e-6),
                  key=lambda p: p.wet_mass)


# --------------------------------------------------------------------------
# Небесные тела системы Кербола (то, что нужно для баллистики)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Body:
    name: str
    mu: float                 # гравитационный параметр, м^3/с^2
    radius: float             # м
    atmosphere_height: float  # м (0 если нет атмосферы)
    soi_radius: float         # м
    rotational_period: float  # с
    orbit_radius: float = 0.0     # радиус орбиты вокруг родителя, м
    orbital_period: float = 0.0   # с
    parent: str | None = None

    @property
    def surface_gravity(self) -> float:
        return self.mu / (self.radius ** 2)

    def circular_speed(self, altitude: float) -> float:
        return (self.mu / (self.radius + altitude)) ** 0.5

    def escape_speed(self, altitude: float) -> float:
        return (2 * self.mu / (self.radius + altitude)) ** 0.5


BODIES: dict[str, Body] = {
    "Kerbol": Body("Kerbol", 1.1723328e18, 261_600_000, 0, float("inf"), 432_000),
    "Kerbin": Body("Kerbin", 3.5316e12, 600_000, 70_000, 84_159_286, 21_549.425,
                   orbit_radius=13_599_840_256, orbital_period=9_203_545, parent="Kerbol"),
    "Mun": Body("Mun", 6.5138398e10, 200_000, 0, 2_429_559.1, 138_984.38,
                orbit_radius=12_000_000, orbital_period=138_984.38, parent="Kerbin"),
    "Minmus": Body("Minmus", 1.7658e9, 60_000, 0, 2_247_428.4, 40_400,
                   orbit_radius=47_000_000, orbital_period=1_077_311, parent="Kerbin"),
    "Duna": Body("Duna", 3.0136321e11, 320_000, 50_000, 47_921_949, 65_517.859,
                 orbit_radius=20_726_155_264, orbital_period=17_315_400, parent="Kerbol"),
}


def get_body(name: str) -> Body:
    if name not in BODIES:
        raise KeyError(f"Небесное тело '{name}' отсутствует в базе KIA")
    return BODIES[name]


# --------------------------------------------------------------------------
# Справочные бюджеты delta-V (м/с) — стандартная карта KSP
# --------------------------------------------------------------------------
DELTA_V_MAP: dict[str, float] = {
    "kerbin_surface_to_lko": 3400.0,
    "lko_to_mun_transfer": 860.0,
    "mun_capture": 310.0,
    "mun_orbit_to_surface": 580.0,
    "mun_surface_to_orbit": 580.0,
    "mun_return_to_kerbin": 310.0,
    "lko_to_minmus_transfer": 930.0,
    "minmus_capture": 160.0,
}


def mission_delta_v(*legs: str, margin: float = 1.10) -> float:
    """Суммарный бюджет delta-V для перечисленных участков с запасом."""
    total = 0.0
    for leg in legs:
        if leg not in DELTA_V_MAP:
            raise KeyError(f"Неизвестный участок миссии: {leg}")
        total += DELTA_V_MAP[leg]
    return total * margin
