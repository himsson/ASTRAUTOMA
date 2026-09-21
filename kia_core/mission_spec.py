"""Постановка задачи для ИИ: что именно требуется от полёта.

MissionSpec — единственный источник целей для Инженера, Пилота и наград.
Создаётся из текста, который оператор пишет в консоли KIA:
    «лети на Муну», «выйди на орбиту 100 км», «слетай на Минмус и вернись».
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import DATA_DIR
from .logging_setup import get_logger

log = get_logger("mission_spec")

SPEC_FILE = DATA_DIR / "mission_spec.json"

# Названия тел: русские и английские варианты -> имя в KSP
BODY_ALIASES = {
    "кербин": "Kerbin", "кербине": "Kerbin", "кербина": "Kerbin", "kerbin": "Kerbin",
    "муна": "Mun", "муну": "Mun", "муне": "Mun", "муны": "Mun", "мун": "Mun",
    "mun": "Mun", "муна.": "Mun",
    "минмус": "Minmus", "минмуса": "Minmus", "минмусе": "Minmus", "minmus": "Minmus",
    "дюна": "Duna", "дюну": "Duna", "дюне": "Duna", "duna": "Duna",
    "кербол": "Kerbol", "солнце": "Kerbol", "kerbol": "Kerbol", "sun": "Kerbol",
}

OBJECTIVE_ALIASES = {
    "orbit": ("орбит", "orbit", "виток"),
    "flyby": ("облёт", "облет", "пролёт", "пролет", "flyby"),
    "land": ("посад", "сядь", "садись", "land", "прилун"),
    "return": ("верн", "домой", "назад", "return", "обратно"),
    "science": ("наук", "science", "эксперимент"),
}


@dataclass
class MissionSpec:
    """Формальная постановка задачи."""
    title: str = "Орбита Муны"
    target_body: str = "Mun"
    home_body: str = "Kerbin"
    park_orbit_altitude: float = 80_000.0     # опорная орбита у дома, м
    target_orbit_altitude: float = 30_000.0   # целевая орбита у цели, м
    objectives: list[str] = field(default_factory=lambda: ["orbit"])
    # KIA строит ТОЛЬКО беспилотные аппараты: управление даёт зондовое ядро,
    # живой экипаж не берётся никогда (нет риска для кербонавтов и нет
    # отказа «Нет управления» при пустой капсуле).
    crewed: bool = False
    collect_science: bool = True
    dv_margin: float = 1.10                   # запас dV сверх расчёта
    max_wall_time: float = 2700.0             # реальных секунд на попытку
    raw_command: str = ""
    # СОЗВЕЗДИЕ СВЯЗИ. Сколько спутников должно стоять на орбите и какое
    # место в цепочке занимает этот. Четыре аппарата через 90° закрывают
    # шар целиком: у каждого всегда есть сосед в прямой видимости.
    constellation_size: int = 0
    constellation_slot: int = 0

    @property
    def constellation_phase(self) -> float:
        """Угол точки стояния в градусах: 0, 90, 180, 270."""
        if self.constellation_size <= 0:
            return 0.0
        return (360.0 / self.constellation_size) * self.constellation_slot

    # ------------------------------------------------------------------
    @property
    def needs_transfer(self) -> bool:
        return self.target_body != self.home_body

    @property
    def needs_landing(self) -> bool:
        return "land" in self.objectives

    @property
    def needs_return(self) -> bool:
        return "return" in self.objectives

    @property
    def is_relay(self) -> bool:
        """Спутник связи, а не исследовательский аппарат.

        Отличие не косметическое: спутнику связи нужна антенна-РЕТРАНСЛЯТОР
        (обычная говорит только с Кербином и чужой сигнал не пересылает),
        питание на работу в тени и собственный двигатель — им спутник
        разводится по своей точке созвездия уже после отделения.
        """
        return "relay" in self.objectives

    @property
    def is_rover(self) -> bool:
        """Луноход: садится и ЕЗДИТ, а не стоит.

        Отличие от посадочного модуля — в том, что остаётся после
        касания. Разбор эталона оператора «Вездеход со спусковым
        модулем»: шесть колёс под палубой, приборы сверху, а всё
        топливо и двигатели спуска — над разделителем и сбрасываются.
        """
        return "rover" in self.objectives

    @property
    def needs_capture(self) -> bool:
        return self.needs_transfer and "flyby" not in self.objectives

    def describe(self) -> str:
        bits = [f"цель: {self.target_body}"]
        if self.needs_transfer:
            bits.append(f"опорная орбита {self.park_orbit_altitude/1000:.0f} км")
            if self.needs_capture:
                bits.append(f"орбита у цели {self.target_orbit_altitude/1000:.0f} км")
        else:
            bits.append(f"орбита {self.park_orbit_altitude/1000:.0f} км")
        if self.needs_landing:
            bits.append("посадка")
        if self.needs_return:
            bits.append("возвращение")
        bits.append("беспилотный аппарат")
        return " | ".join(bits)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MissionSpec":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or SPEC_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MissionSpec":
        p = Path(path or SPEC_FILE)
        if p.exists():
            try:
                return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Не удалось прочитать задачу (%s) — беру задачу по умолчанию", exc)
        return cls()


# ==========================================================================
def parse_command(text: str, base: MissionSpec | None = None) -> MissionSpec:
    """Строит задачу из фразы оператора.

    Понимает: «лети на Муну», «выйди на орбиту 100 км», «сядь на Мун и вернись»,
    «облёт Минмуса без экипажа», «орбита Кербина 250км».
    """
    spec = MissionSpec.from_dict((base or MissionSpec()).to_dict())
    spec.raw_command = text.strip()
    lowered = text.lower().replace("ё", "е")

    # --- целевое тело ---
    target = None
    for word in re.findall(r"[а-яa-z]+", lowered):
        alias = BODY_ALIASES.get(word)
        if alias:
            target = alias
            break
    if target:
        spec.target_body = target

    # --- цели ---
    objectives: list[str] = []
    for key, needles in OBJECTIVE_ALIASES.items():
        if any(n in lowered for n in needles):
            objectives.append(key)
    if not objectives:
        objectives = ["orbit"]
    if "land" in objectives and "orbit" not in objectives:
        objectives.append("orbit")
    spec.objectives = objectives
    spec.collect_science = "science" in objectives or spec.collect_science

    # --- высота орбиты ---
    altitudes = []
    for value, unit in re.findall(r"(\d+(?:[.,]\d+)?)\s*(км|km|м\b|m\b)?", lowered):
        number = float(value.replace(",", "."))
        if unit in ("км", "km") or (not unit and number < 1000):
            number *= 1000.0
        altitudes.append(number)
    altitudes = [a for a in altitudes if 5_000 <= a <= 5_000_000]
    if altitudes:
        if target is None and "орбит" in lowered:
            # «выйди на орбиту 100 км» без названия тела — орбита у дома
            spec.target_body = spec.home_body
            spec.park_orbit_altitude = altitudes[0]
        elif spec.target_body == spec.home_body:
            spec.park_orbit_altitude = altitudes[0]
        elif len(altitudes) >= 2:
            spec.park_orbit_altitude, spec.target_orbit_altitude = altitudes[0], altitudes[1]
        else:
            spec.target_orbit_altitude = altitudes[0]

    # --- экипаж ---
    # Пилотируемые запуски отключены на уровне архитектуры: KIA конструирует
    # исключительно беспилотные аппараты.
    spec.crewed = False
    if any(w in lowered for w in ("экипаж", "кербонавт", "с людьми", "crewed")):
        log.info("Запрос на экипаж проигнорирован: KIA строит только беспилотники")

    spec.title = _title_for(spec)
    log.info("Задача разобрана: %s", spec.describe())
    return spec


def _title_for(spec: MissionSpec) -> str:
    if spec.target_body == spec.home_body:
        base = f"Орбита {spec.park_orbit_altitude/1000:.0f} км"
    elif spec.needs_landing:
        base = f"Посадка на {spec.target_body}"
    elif not spec.needs_capture:
        base = f"Облёт {spec.target_body}"
    else:
        base = f"Орбита {spec.target_body}"
    if spec.needs_return:
        base += " с возвращением"
    return base
