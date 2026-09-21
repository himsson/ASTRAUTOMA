"""Глобальная конфигурация Kerbal Intelligence Agency.

Все модули берут параметры отсюда. Значения можно переопределить
переменными окружения (KIA_*) или файлом config.local.json в корне проекта.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

G0 = 9.80665  # стандартное ускорение свободного падения, м/с^2


@dataclass
class ConnectionConfig:
    name: str = "KerbalIntelligenceAgency"
    address: str = "127.0.0.1"
    rpc_port: int = 50000
    stream_port: int = 50001
    connect_timeout: float = 10.0
    retry_attempts: int = 3
    retry_delay: float = 3.0


@dataclass
class MissionConfig:
    """Параметры целевой миссии текущего этапа: Кербин -> орбита Муны."""
    target_body: str = "Mun"
    home_body: str = "Kerbin"
    park_orbit_altitude: float = 80_000.0      # целевая круговая орбита у Кербина, м
    park_orbit_tolerance: float = 5_000.0      # допуск по апо/периапсису
    mun_capture_periapsis: float = 30_000.0    # желаемый периапсис у Муны, м
    max_mission_time: float = 3600.0 * 6       # игровых секунд на попытку
    max_wall_time: float = 60.0 * 45           # реальных секунд на попытку
    min_electric_charge: float = 5.0


@dataclass
class CraftConfig:
    """Где лежат сгенерированные .craft файлы KSP."""
    ksp_root: str = os.environ.get("KIA_KSP_ROOT", "")
    save_name: str = os.environ.get("KIA_SAVE_NAME", "default")
    craft_name: str = "KIA_Autogen"
    # Если False — система не пишет .craft, а работает с уже активным судном.
    generate_craft_files: bool = True
    launch_from_vab: bool = True
    # ЖЁСТКАЯ ПРИВЯЗКА К ЧУЖОМУ ЧЕРТЕЖУ.
    #
    # Когда здесь стоит имя, конструктор отключается целиком: KIA не
    # строит и не пишет свои ракеты, а раз за разом поднимает ИМЕННО
    # этот чертёж из ангара и учится летать на нём. Эволюция при этом
    # продолжает крутить полётные гены — профиль подъёма, момент
    # разворота, — потому что учиться теперь надо не строить, а
    # управлять.
    fixed_craft: str = os.environ.get("KIA_FIXED_CRAFT", "")


@dataclass
class LearningConfig:
    population_size: int = 8
    elite_count: int = 2
    mutation_rate: float = 0.35
    mutation_scale: float = 0.18
    crossover_rate: float = 0.6
    max_generations: int = 50
    random_seed: int | None = None
    history_file: str = str(DATA_DIR / "history.json")
    best_genome_file: str = str(DATA_DIR / "best_genome.json")


@dataclass
class ControlConfig:
    physics_tick: float = 0.05          # шаг основного цикла управления, с
    stream_rate: float = 10.0           # Гц для потоков телеметрии
    autostage_thrust_ratio: float = 0.1  # порог падения тяги для отстрела ступени
    max_time_warp: int = 4


@dataclass
class Config:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    craft: CraftConfig = field(default_factory=CraftConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_overrides(cfg: Config) -> Config:
    local = ROOT / "config.local.json"
    if local.exists():
        raw = json.loads(local.read_text(encoding="utf-8"))
        for section, values in raw.items():
            target = getattr(cfg, section, None)
            if target is None or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
    # Переменные окружения вида KIA_CONNECTION_ADDRESS
    for section_name in cfg.to_dict():
        target = getattr(cfg, section_name)
        for key in vars(target):
            env_key = f"KIA_{section_name.upper()}_{key.upper()}"
            if env_key in os.environ:
                current = getattr(target, key)
                raw = os.environ[env_key]
                try:
                    if isinstance(current, bool):
                        value = raw.strip().lower() in ("1", "true", "yes", "on")
                    elif isinstance(current, int):
                        value = int(raw)
                    elif isinstance(current, float):
                        value = float(raw)
                    else:
                        value = raw
                except ValueError:
                    continue
                setattr(target, key, value)
    return cfg


CONFIG = _apply_overrides(Config())
