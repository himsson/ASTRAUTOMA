"""Каталог деталей, читаемый напрямую из установки KSP.

Никаких зашитых таблиц: сканируются файлы GameData/**/*.cfg, из них
извлекаются реальные параметры деталей (масса, тяга, Isp, стыковочные ноды,
ресурсы). Это даёт две вещи:
  * корректные имена деталей для .craft (главная причина прошлых поломок);
  * настоящую свободу конструктору — он видит всё, что есть в игре, включая моды.

Результат кэшируется в data/parts_cache.json и переиспользуется, пока
состав GameData не изменился.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import CONFIG, DATA_DIR
from ..logging_setup import get_logger
from .config_node import ConfigNode, localized, parse_file

log = get_logger("engineer.catalog")

CACHE_FILE = DATA_DIR / "parts_cache.json"

# Плотности на случай, если файлы ресурсов недоступны (т на единицу)
FALLBACK_DENSITY = {
    "LiquidFuel": 0.005, "Oxidizer": 0.005, "SolidFuel": 0.0075,
    "MonoPropellant": 0.004, "XenonGas": 0.0001, "ElectricCharge": 0.0,
    "Ore": 0.010, "Ablator": 0.001,
}

ENGINE_MODULES = {"ModuleEngines", "ModuleEnginesFX", "ModuleEnginesRF"}
DECOUPLER_MODULES = {"ModuleDecouple", "ModuleAnchoredDecoupler"}
COMMAND_MODULES = {"ModuleCommand"}
PARACHUTE_MODULES = {"ModuleParachute"}
SOLAR_MODULES = {"ModuleDeployableSolarPanel"}
SCIENCE_MODULES = {"ModuleScienceExperiment"}
ANTENNA_MODULES = {"ModuleDataTransmitter"}
# Отсеки с полостью: приборы внутри не обдуваются потоком. Обтекатель
# (`ModuleProceduralFairing`) сюда НЕ входит — он раскрывается и работает
# иначе, ему отдельная обработка.
CARGO_BAY_MODULES = {"ModuleCargoBay", "ModuleServiceModule"}

# Размеры стыковочных узлов KSP: индекс -> диаметр, м
NODE_SIZE_M = {0: 0.625, 1: 1.25, 2: 2.5, 3: 3.75, 4: 5.0}

# Диаметр обшивки по имени профиля из `bulkheadProfiles`. Профили без
# круглого сечения (mk2, mk3, srf, size0p5 и прочие) сюда не входят:
# для них диаметр по узлу — не худшая оценка.
BULKHEAD_SIZE_M = {
    "size0": 0.625,
    "size1": 1.25,
    "size1p5": 1.875,
    "size2": 2.5,
    "size3": 3.75,
    "size4": 5.0,
}


def adapter_ends(info: "PartInfo") -> tuple[float, float]:
    """Диаметры верхнего и нижнего торцов переходника.

    Различаются по размеру узла из cfg. У части деталей Making History
    оба узла объявлены ОДНИМ размером (у FL-A151S оба `size = 2`), и
    тогда по ним не понять, какой конец толще. В этом случае считаем, что
    широкий конец внизу: ракета шире у основания, и переходник в стеке
    стоит именно так.
    """
    sizes = info.bulkhead_sizes
    if len(sizes) < 2:
        d = info.hull_diameter
        return d, d
    top, bottom = info.top_node, info.bottom_node
    if top is not None and bottom is not None and top.size != bottom.size:
        td = NODE_SIZE_M.get(top.size)
        bd = NODE_SIZE_M.get(bottom.size)
        if td in sizes and bd in sizes:
            return td, bd
        return ((sizes[0], sizes[-1]) if top.size < bottom.size
                else (sizes[-1], sizes[0]))
    return sizes[0], sizes[-1]


@dataclass
class AttachNode:
    name: str            # top / bottom / bottom01 / ...
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    size: int = 1

    def to_dict(self) -> dict:
        return {"name": self.name, "position": list(self.position),
                "orientation": list(self.orientation), "size": self.size}

    @classmethod
    def from_dict(cls, d: dict) -> "AttachNode":
        return cls(d["name"], tuple(d["position"]), tuple(d["orientation"]),
                   int(d.get("size", 1)))


@dataclass
class PartInfo:
    """Всё, что KIA знает о детали — прочитано из cfg игры."""
    name: str                       # имя из cfg (например, fuelTank_long)
    title: str
    category: str
    cfg_path: str
    dry_mass: float = 0.0           # т
    cost: float = 0.0
    crash_tolerance: float = 6.0
    rescale: float = 1.0
    crew_capacity: int = 0
    modules: list[str] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)   # имя -> maxAmount
    stack_nodes: list[AttachNode] = field(default_factory=list)
    surface_node: tuple[float, float, float] | None = None
    # Направление узла поверхностного крепления из cfg (node_attach, три
    # последних числа). Само по себе оно НЕ говорит, куда деталь смотрит
    # наружу, — см. `outward_axis` в craft_writer.
    surface_node_orient: tuple[float, float, float] | None = None
    # Поворачивается ли панель к солнцу и сколько даёт тока. Неподвижная
    # накладка заряжает только при удачном развороте всего аппарата.
    sun_tracking: bool = False
    charge_rate: float = 0.0
    # Дальность антенны из cfg (`antennaPower`), метры. Нужна, чтобы
    # брать связь с запасом, а не самую лёгкую: у «Коммунотрона 16» это
    # 500 км, у DTS-M1 — 2 Гм, разница в четыре тысячи раз.
    antenna_power: float = 0.0
    # РЕТРАНСЛЯТОР ИЛИ ПРОСТО АНТЕННА. Обычная антенна говорит только с
    # Кербином; ретранслятор (`antennaType = RELAY`) пересылает чужой
    # сигнал, и без него спутник связи бессмысленен. Замер по эталону
    # оператора «Спутник связи SWM-94»: там стоит RelayAntenna100.
    is_relay: bool = False
    # Управляющий момент маховика, кН·м. Замер живого полёта: по тангажу
    # и рысканью аппарат имел 31.77, а по КРЕНУ всего 0.50 — качание
    # сопла крен не даёт вовсе. Его и не хватало: 41°/с вращения, ошибка
    # автопилота 92°, и погасить нечем.
    roll_torque: float = 0.0
    can_surface_attach: bool = True
    # двигатель
    max_thrust: float = 0.0         # кН (вакуум)
    isp_vac: float = 0.0
    isp_asl: float = 0.0
    propellants: list[str] = field(default_factory=list)
    gimbal: float = 0.0
    tags: str = ""
    bulkhead_profiles: str = ""

    # ---------------- классификация ----------------
    @property
    def craft_id_name(self) -> str:
        """Имя детали так, как его пишет KSP в .craft: '_' заменён на '.'."""
        return self.name.replace("_", ".")

    @property
    def is_engine(self) -> bool:
        return bool(ENGINE_MODULES.intersection(self.modules)) and self.max_thrust > 0

    @property
    def is_liquid_engine(self) -> bool:
        """Ракетный ЖРД на LF+Ox: без воздухозаборника, с окислителем."""
        if not self.is_engine:
            return False
        props = set(self.propellants)
        return "LiquidFuel" in props and "Oxidizer" in props and "IntakeAir" not in props

    @property
    def is_air_breathing(self) -> bool:
        return "IntakeAir" in self.propellants

    @property
    def is_solid_booster(self) -> bool:
        return self.is_engine and "SolidFuel" in self.propellants

    @property
    def is_tank(self) -> bool:
        if self.is_engine:
            return False
        return (self.resources.get("LiquidFuel", 0) > 0
                and self.resources.get("Oxidizer", 0) > 0)

    @property
    def is_stack_tank(self) -> bool:
        """Обычный цилиндрический бак для вертикальной сборки."""
        if not self.is_tank or self.category != "FuelTank":
            return False
        if self.top_node is None or self.bottom_node is None:
            return False
        if self.top_node.size != self.bottom_node.size:
            return False
        lowered = self.name.lower()
        if any(tag in lowered for tag in ("mk2", "mk3", "adapter", "toroid")):
            return False
        # верх и низ должны лежать на одной оси
        return abs(self.top_node.position[0]) < 1e-6 and abs(self.bottom_node.position[0]) < 1e-6

    @property
    def is_decoupler(self) -> bool:
        return (bool(DECOUPLER_MODULES.intersection(self.modules))
                and self.category == "Coupling")

    @property
    def is_command(self) -> bool:
        return bool(COMMAND_MODULES.intersection(self.modules))

    @property
    def is_crewed(self) -> bool:
        return self.crew_capacity > 0

    @property
    def is_parachute(self) -> bool:
        return bool(PARACHUTE_MODULES.intersection(self.modules))

    @property
    def is_solar(self) -> bool:
        return bool(SOLAR_MODULES.intersection(self.modules))

    @property
    def is_battery(self) -> bool:
        return (self.resources.get("ElectricCharge", 0) > 0
                and not self.is_command and not self.is_solar)

    @property
    def is_science(self) -> bool:
        return bool(SCIENCE_MODULES.intersection(self.modules))

    @property
    def is_antenna(self) -> bool:
        return bool(ANTENNA_MODULES.intersection(self.modules))

    # ---------------- геометрия ----------------
    def node(self, name: str) -> AttachNode | None:
        """Узел по имени, с учётом нумерованных вариантов.

        ИМЕНА УЗЛОВ В ИГРЕ НЕ КАНОНИЧНЫ. У носового обтекателя нижний
        узел зовётся `bottom01`, а не `bottom`, и точный поиск его не
        находил. Сборщик считал, что узла нет, брал запасную оценку — и
        конус садился внутрь корпуса вместо макушки. Оператор увидел это
        на снимке из игры раньше, чем показал любой расчёт.
        """
        for n in self.stack_nodes:
            if n.name == name:
                return n
        # `bottom` ищем среди bottom01, bottom02...; так же и `top`
        for n in self.stack_nodes:
            if n.name.startswith(name) and n.name[len(name):].isdigit():
                return n
        return None

    @property
    def top_node(self) -> AttachNode | None:
        return self.node("top") or self.node("top01")

    @property
    def bottom_node(self) -> AttachNode | None:
        return self.node("bottom") or self.node("bottom01")

    @property
    def diameter(self) -> float:
        """Диаметр стыковочного узла (по нижнему, иначе по верхнему)."""
        node = self.bottom_node or self.top_node
        if node is None:
            return 0.0
        return NODE_SIZE_M.get(node.size, 1.25)

    @property
    def bulkhead_sizes(self) -> list[float]:
        """Все диаметры, которыми деталь стыкуется, по возрастанию.

        У обычного бака он один, у переходника — два: `size1p5, size1`
        значит «сверху 1.875 м, снизу 1.25 м» (или наоборот). По этому
        полю и подбирается переходник между баками разной толщины.
        """
        out = set()
        for token in self.bulkhead_profiles.replace(" ", "").split(","):
            size = BULKHEAD_SIZE_M.get(token.lower())
            if size:
                out.add(size)
        return sorted(out)

    @property
    def is_cargo_bay(self) -> bool:
        """Грузовой (служебный) отсек: закрытая оболочка с полостью внутри.

        Приборы, поставленные внутрь, не участвуют в обтекании — ради
        этого отсек и берётся.
        """
        return bool(CARGO_BAY_MODULES.intersection(self.modules))

    @property
    def hull_diameter(self) -> float:
        """Диаметр САМОГО КОРПУСА, а не стыковочного узла.

        Это разные вещи, и разница видна в игре. У баков Making History
        размера 1p5 в cfg стоит узел размера 2:

            node_stack_top   = 0, 0.234375, 0, 0, 1, 0, 2
            bulkheadProfiles = size1p5, srf

        Узел такой ради совместимости стыковки, а обшивка у детали —
        1.875 м. Пока радиальное крепление считало по узлу, оперение
        садилось на радиус 1.250 при настоящем 0.9375 и висело в
        воздухе в тридцати сантиметрах от борта — видно на снимке из
        игры.

        Для подбора двигателей и баков по-прежнему берётся `diameter`
        (по узлу): там важна именно стыкуемость.
        """
        for token in self.bulkhead_profiles.replace(" ", "").split(","):
            size = BULKHEAD_SIZE_M.get(token.lower())
            if size:
                return size
        return self.diameter

    @property
    def height(self) -> float:
        top = self.top_node
        bottom = self.bottom_node
        if top and bottom:
            return abs(top.position[1] - bottom.position[1])
        return 0.5

    # ---------------- массы и тяга ----------------
    def resource_mass(self, densities: dict[str, float]) -> float:
        return sum(amount * densities.get(name, FALLBACK_DENSITY.get(name, 0.0))
                   for name, amount in self.resources.items())

    def wet_mass(self, densities: dict[str, float]) -> float:
        return self.dry_mass + self.resource_mass(densities)

    def fuel_units(self) -> float:
        return self.resources.get("LiquidFuel", 0.0) + self.resources.get("Oxidizer", 0.0)

    def thrust_at(self, pressure_atm: float) -> float:
        """Тяга при заданном давлении: F = F_vac * Isp(p)/Isp_vac."""
        if self.isp_vac <= 0:
            return self.max_thrust
        return self.max_thrust * (self.isp_at(pressure_atm) / self.isp_vac)

    def isp_at(self, pressure_atm: float) -> float:
        p = max(0.0, min(1.0, pressure_atm))
        return self.isp_vac + (self.isp_asl - self.isp_vac) * p

    # ---------------- сериализация ----------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["stack_nodes"] = [n.to_dict() for n in self.stack_nodes]
        d["surface_node"] = list(self.surface_node) if self.surface_node else None
        d["surface_node_orient"] = (list(self.surface_node_orient)
                                    if self.surface_node_orient else None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PartInfo":
        d = dict(d)
        d["stack_nodes"] = [AttachNode.from_dict(n) for n in d.get("stack_nodes", [])]
        sn = d.get("surface_node")
        d["surface_node"] = tuple(sn) if sn else None
        so = d.get("surface_node_orient")
        d["surface_node_orient"] = tuple(so) if so else None
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def describe(self) -> str:
        bits = [f"{self.title} [{self.name}]", f"{self.dry_mass:.3f} т"]
        if self.is_engine:
            bits.append(f"{self.max_thrust:.0f} кН, Isp {self.isp_asl:.0f}/{self.isp_vac:.0f} с")
        if self.is_tank:
            bits.append(f"LF {self.resources.get('LiquidFuel', 0):.0f} + "
                        f"Ox {self.resources.get('Oxidizer', 0):.0f}")
        if self.diameter:
            bits.append(f"⌀{self.diameter:.3g} м")
        return " | ".join(bits)


# ==========================================================================
class PartCatalog:
    """Каталог всех деталей текущей установки KSP."""

    def __init__(self, parts: dict[str, PartInfo] | None = None,
                 densities: dict[str, float] | None = None,
                 source: str = ""):
        self.parts: dict[str, PartInfo] = parts or {}
        self.densities: dict[str, float] = densities or dict(FALLBACK_DENSITY)
        self.source = source

    # ---------------- сканирование ----------------
    @classmethod
    def scan(cls, gamedata: Path) -> "PartCatalog":
        t0 = time.time()
        parts: dict[str, PartInfo] = {}
        densities: dict[str, float] = dict(FALLBACK_DENSITY)
        cfg_files = list(gamedata.rglob("*.cfg"))

        for cfg in cfg_files:
            try:
                root = parse_file(cfg)
            except Exception as exc:
                log.debug("Не разобран %s: %s", cfg.name, exc)
                continue
            for node in root.children:
                if node.name == "PART":
                    info = _part_from_node(node, cfg)
                    if info is not None:
                        parts[info.name] = info
                elif node.name == "RESOURCE_DEFINITION":
                    name = node.get("name")
                    if name:
                        densities[name] = node.get_float("density", 0.0)

        log.info("Каталог KSP: %d деталей из %d cfg за %.1f с",
                 len(parts), len(cfg_files), time.time() - t0)
        return cls(parts, densities, source=str(gamedata))

    # ---------------- кэш ----------------
    @classmethod
    def load(cls, gamedata: Path | None = None, use_cache: bool = True) -> "PartCatalog":
        """Возвращает каталог: из кэша, из GameData или встроенный резервный."""
        gamedata = gamedata or _find_gamedata()
        if gamedata is None:
            log.warning("GameData не найдена — используется встроенный мини-каталог")
            return cls.fallback()

        signature = _signature(gamedata)
        if use_cache and CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if data.get("signature") == signature:
                    parts = {name: PartInfo.from_dict(d)
                             for name, d in data["parts"].items()}
                    log.info("Каталог загружен из кэша: %d деталей", len(parts))
                    return cls(parts, data.get("densities", FALLBACK_DENSITY),
                               source=data.get("source", str(gamedata)))
            except Exception as exc:
                log.debug("Кэш каталога не прочитан: %s", exc)

        catalog = cls.scan(gamedata)
        catalog.save_cache(signature)
        return catalog

    def save_cache(self, signature: str = "") -> None:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({
            "signature": signature,
            "source": self.source,
            "densities": self.densities,
            "parts": {n: p.to_dict() for n, p in self.parts.items()},
        }, ensure_ascii=False), encoding="utf-8")
        log.debug("Кэш каталога записан: %s", CACHE_FILE)

    @classmethod
    def fallback(cls) -> "PartCatalog":
        """Мини-каталог из встроенной базы — чтобы система работала без KSP."""
        from .parts_db import ALL_PARTS
        parts = {}
        for p in ALL_PARTS.values():
            info = PartInfo(
                name=p.part_id, title=p.title, category=p.category.value,
                cfg_path="<builtin>", dry_mass=p.dry_mass, cost=p.cost,
                crash_tolerance=p.crash_tolerance,
                modules=(["ModuleEngines"] if p.thrust_vac > 0 else []),
                resources={k: v for k, v in
                           (("LiquidFuel", p.liquid_fuel), ("Oxidizer", p.oxidizer),
                            ("ElectricCharge", p.electric_charge)) if v},
                max_thrust=p.thrust_vac, isp_vac=p.isp_vac, isp_asl=p.isp_asl,
                propellants=(["LiquidFuel", "Oxidizer"] if p.thrust_vac > 0 else []),
                gimbal=p.gimbal,
            )
            size_idx = {0.625: 0, 1.25: 1, 2.5: 2, 3.75: 3}.get(p.size, 1)
            half = 0.5
            info.stack_nodes = [
                AttachNode("top", (0.0, half, 0.0), (0.0, 1.0, 0.0), size_idx),
                AttachNode("bottom", (0.0, -half, 0.0), (0.0, -1.0, 0.0), size_idx),
            ]
            if p.category.value in ("command",):
                info.modules.append("ModuleCommand")
                info.crew_capacity = 1
            if p.category.value == "parachute":
                info.modules.append("ModuleParachute")
            if p.category.value == "decoupler":
                info.modules.append("ModuleDecouple")
            if p.category.value == "science":
                info.modules.append("ModuleScienceExperiment")
            parts[p.part_id] = info
        return cls(parts, dict(FALLBACK_DENSITY), source="<builtin>")

    # ---------------- выборки ----------------
    def get(self, name: str) -> PartInfo | None:
        return self.parts.get(name)

    def all(self) -> list[PartInfo]:
        return list(self.parts.values())

    def engines(self, diameter: float | None = None, liquid_only: bool = True,
                max_mass: float | None = None) -> list[PartInfo]:
        out = []
        for p in self.parts.values():
            if not p.is_engine or p.category != "Engine":
                continue
            if liquid_only and not p.is_liquid_engine:
                continue
            if p.top_node is None:          # двигателю нужен верхний узел под бак
                continue
            if diameter is not None and abs(p.diameter - diameter) > 1e-6:
                continue
            if max_mass is not None and p.dry_mass > max_mass:
                continue
            out.append(p)
        return sorted(out, key=lambda p: p.max_thrust)

    def tanks(self, diameter: float | None = None) -> list[PartInfo]:
        out = [p for p in self.parts.values() if p.is_stack_tank
               and (diameter is None or abs(p.diameter - diameter) < 1e-6)]
        return sorted(out, key=lambda p: p.fuel_units())

    def decouplers(self, diameter: float | None = None) -> list[PartInfo]:
        out = [p for p in self.parts.values() if p.is_decoupler
               and len(p.stack_nodes) >= 2 and p.top_node and p.bottom_node
               and (diameter is None or abs(p.diameter - diameter) < 1e-6)]
        return sorted(out, key=lambda p: p.dry_mass)

    def pods(self, crewed: bool | None = None, diameter: float | None = None) -> list[PartInfo]:
        out = [p for p in self.parts.values()
               if p.is_command and p.category == "Pods" and p.bottom_node
               and (crewed is None or p.is_crewed == crewed)
               and (diameter is None or abs(p.diameter - diameter) < 1e-6)]
        return sorted(out, key=lambda p: p.dry_mass)

    def parachutes(self, diameter: float | None = None) -> list[PartInfo]:
        out = [p for p in self.parts.values() if p.is_parachute]
        if diameter is not None:
            sized = [p for p in out if abs(p.diameter - diameter) < 1e-6]
            if sized:
                out = sized
        return sorted(out, key=lambda p: p.dry_mass)

    def batteries(self) -> list[PartInfo]:
        return sorted((p for p in self.parts.values() if p.is_battery),
                      key=lambda p: p.dry_mass)

    def solar_panels(self) -> list[PartInfo]:
        return sorted((p for p in self.parts.values() if p.is_solar),
                      key=lambda p: p.dry_mass)

    def oriented_adapter(self, top_d: float, bottom_d: float) -> PartInfo | None:
        """Переходник, у которого верх уже нужного диаметра, а низ — тоже.

        Перевороты вслепую здесь запрещены: попытка развернуть деталь
        «как надо» однажды собрала стек вверх ногами — баки верхней
        ступени встали выше командного модуля. Если детали нужной
        ориентации нет, честнее оставить уступ.
        """
        for part in self.adapters_between(top_d, bottom_d):
            ends = adapter_ends(part)
            if abs(ends[0] - top_d) < 1e-6 and abs(ends[1] - bottom_d) < 1e-6:
                return part
        return None

    def adapters_between(self, d1: float, d2: float) -> list[PartInfo]:
        """Все переходники между двумя диаметрами, лёгкие и топливные первыми."""
        if abs(d1 - d2) < 1e-6:
            return []
        want = tuple(sorted((round(d1, 4), round(d2, 4))))
        fits = [p for p in self.parts.values()
                if not p.is_engine and p.top_node and p.bottom_node
                and tuple(round(x, 4) for x in p.bulkhead_sizes) == want]
        return sorted(fits, key=lambda p: (not p.resources.get("LiquidFuel"),
                                           p.dry_mass))

    def adapter_between(self, d1: float, d2: float) -> PartInfo | None:
        """Переходник между двумя диаметрами, самый лёгкий из подходящих.

        Предпочтение — топливному: он не мёртвый груз, а часть бака.
        Именно такие в игре и зовутся FL-A150 / FL-A151S.
        """
        if abs(d1 - d2) < 1e-6:
            return None
        want = tuple(sorted((round(d1, 4), round(d2, 4))))
        fits = [p for p in self.parts.values()
                if not p.is_engine and p.top_node and p.bottom_node
                and tuple(round(x, 4) for x in p.bulkhead_sizes) == want]
        if not fits:
            return None
        return min(fits, key=lambda p: (not p.resources.get("LiquidFuel"),
                                        p.dry_mass))

    def antennas(self) -> list[PartInfo]:
        return sorted((p for p in self.parts.values() if p.is_antenna),
                      key=lambda p: p.dry_mass)

    def science(self) -> list[PartInfo]:
        return sorted((p for p in self.parts.values() if p.is_science),
                      key=lambda p: p.dry_mass)

    def probe_cores(self, diameter: float | None = None) -> list[PartInfo]:
        """Беспилотные командные модули (Probe Core / OKTO / Stayputnik).

        KIA строит только беспилотные аппараты, поэтому наверх ставится
        зондовое ядро категории Command: оно даёт управление и SAS без
        живого экипажа. Требуется нижний стыковочный узел и собственный
        запас электричества (иначе аппарат неуправляем).
        """
        out = []
        for p in self.parts.values():
            if not p.is_command or p.crew_capacity > 0:
                continue
            if p.category not in ("Pods", "Control"):
                continue
            if p.bottom_node is None:
                continue
            if diameter is not None and abs(p.diameter - diameter) > 1e-6:
                continue
            out.append(p)
        # Порядок важен для управляемости:
        #   1) есть маховик (ModuleReactionWheel) — иначе автопилоту нечем
        #      разворачивать аппарат и курс держать нечем;
        #   2) есть собственный запас энергии;
        #   3) минимальная масса.
        return sorted(out, key=lambda p: ("ModuleReactionWheel" not in p.modules,
                                          p.resources.get("ElectricCharge", 0.0) <= 0,
                                          p.dry_mass))

    def reaction_wheels(self, diameter: float | None = None) -> list[PartInfo]:
        """Отдельные маховики — то, чем аппарат разворачивают без топлива."""
        out = [p for p in self.parts.values()
               if "ModuleReactionWheel" in p.modules and not p.is_command
               and p.top_node and p.bottom_node
               and (diameter is None or abs(p.diameter - diameter) < 1e-6)]
        return sorted(out, key=lambda p: p.dry_mass)

    def rcs_blocks(self) -> list[PartInfo]:
        """Блоки реактивной системы управления для поверхностной навески."""
        out = [p for p in self.parts.values()
               if ("ModuleRCSFX" in p.modules or "ModuleRCS" in p.modules)
               and p.can_surface_attach and p.dry_mass <= 0.06]
        return sorted(out, key=lambda p: p.dry_mass)

    def launch_clamps(self) -> list[PartInfo]:
        """Пусковые мачты: держат ракету на столе до зажигания.

        Зачем они вообще. Высокая ракета малого диаметра на стартовом
        столе стоит на одном сочленении двигателя с площадкой, и в KSP
        этого хватает, чтобы она качнулась и завалилась ЕЩЁ ДО СТАРТА.
        Никакой пилот тут не поможет: тренажёр такого отказа не знает
        вовсе — там аппарат на столе удерживается вертикально до отрыва
        (`ground_contact`), поэтому сеть подобного просто не видела и
        научиться этому не может. Это задача конструктора, а не пилота.

        Мачта отстреливается на зажигании нижней ступени, весит 0.1 т и
        в полёте не участвует.
        """
        out = [p for p in self.parts.values()
               if "LaunchClamp" in p.modules
               or p.name.lower().startswith("launchclamp")]
        return sorted(out, key=lambda p: p.dry_mass)

    def nose_cones(self, diameter: float | None = None) -> list[PartInfo]:
        """Носовые обтекатели: снимают лобовое сопротивление с макушки.

        В стоковой аэродинамике KSP тупой торец наверху стоит дорого:
        деталь без обтекателя набирает полный коэффициент сопротивления,
        и ракета теряет сотни м/с на подъёме. Конус закрывает торец и
        отдаёт эти метры обратно.

        Признак конуса: есть НИЖНИЙ узел и нет верхнего — то есть деталь
        замыкает стопку сверху. Крыльев и рулей это не касается: у них
        стыковочных узлов нет вовсе, только поверхностное крепление.
        """
        out = []
        for p in self.parts.values():
            if p.bottom_node is None or p.top_node is not None:
                continue
            if p.category not in ("Aero", "Structural", "Payload"):
                continue
            if p.is_engine or p.is_parachute or p.is_command:
                continue
            if "ModuleControlSurface" in p.modules:
                continue
            if "IntakeAir" in p.resources or "intake" in p.name.lower():
                continue
            if diameter is not None and abs(p.diameter - diameter) > 1e-6:
                continue
            out.append(p)
        return sorted(out, key=lambda p: p.dry_mass)

    def radial_decouplers(self) -> list[PartInfo]:
        """Разделители бокового крепления (TT-38K, TT-70).

        Отличие от стековых: крепятся к борту поверхностно, а несут
        деталь на одном стыковочном узле. Именно ими навешиваются
        радиальные ускорители.
        """
        out = [p for p in self.parts.values()
               if p.is_decoupler and p.can_surface_attach
               and len(p.stack_nodes) <= 1]
        return sorted(out, key=lambda p: p.dry_mass)

    def solid_boosters(self) -> list[PartInfo]:
        """Твердотопливные ускорители для боковой навески.

        Дают много тяги за малую сухую массу и не требуют ни бака, ни
        топливопровода — поэтому именно с них начинается любая тяжёлая
        ракета. Тяга не регулируется и выключить их нельзя: ускоритель
        горит до конца, после чего его сбрасывают.
        """
        out = [p for p in self.parts.values()
               if p.is_solid_booster and p.can_surface_attach
               and p.resources.get("SolidFuel", 0) > 0
               # Категория Engine отсекает башню аварийного спасения
               # (LaunchEscapeSystem, Utility) — она тоже на твёрдом
               # топливе, но тянет капсулу ВВЕРХ, а не ракету на орбиту.
               and p.category == "Engine"
               # Нулевой диаметр — у сепаратрона: это отводящий двигатель
               # на 8 единиц топлива, стыковочных узлов у него нет вовсе.
               and p.diameter > 0]
        return sorted(out, key=lambda p: p.max_thrust)

    # Ракетные стабилизаторы отличаются от самолётных крыльев не модулем, а
    # назначением: их вешают на борт ракеты вертикально, и они компактны.
    # Крыло же рассчитано на фюзеляж в ангаре, торчит вбок на метры и на
    # ракете цепляет пусковые мачты. Замер: конструктор, которому разрешили
    # брать «оперение побольше», поставил Swept Wing Type A — и аппарат
    # взорвался на 68 метрах, счёт -850.
    #
    # Различить их в каталоге больше нечем: ни площади, ни габаритов в cfg
    # нет. Поэтому — по назначению, как его понимает сама игра.
    FIN_WORDS = ("fin", "winglet", "оперен", "стабилиз")
    WING_WORDS = ("wing connector", "strake", "delta wing", "swept wing",
                  "structural wing", "aeroplane", "shuttle")

    def _is_rocket_fin(self, part: PartInfo) -> bool:
        text = f"{part.name} {part.title}".lower()
        if any(word in text for word in self.WING_WORDS):
            return False
        return any(word in text for word in self.FIN_WORDS)

    def rocket_fins(self, steerable: bool | None = None) -> list[PartInfo]:
        """Только стабилизаторы, пригодные для борта ракеты."""
        out = [p for p in self.fins(steerable=steerable) if self._is_rocket_fin(p)]
        return out or self.fins(steerable=steerable)

    def fins(self, steerable: bool | None = None) -> list[PartInfo]:
        """Аэродинамические поверхности для поверхностного крепления."""
        out = []
        for p in self.parts.values():
            if p.category != "Aero" or not p.can_surface_attach:
                continue
            if not p.surface_node or p.dry_mass > 0.15:
                continue
            controls = "ModuleControlSurface" in p.modules
            if "ModuleLiftingSurface" not in p.modules and not controls:
                continue
            if steerable is not None and controls != steerable:
                continue
            out.append(p)
        return sorted(out, key=lambda p: p.dry_mass)

    def diameters(self) -> list[float]:
        return sorted({p.diameter for p in self.parts.values() if p.diameter > 0})

    def stats(self) -> dict:
        return {
            "source": self.source,
            "total": len(self.parts),
            "engines": len(self.engines(liquid_only=False)),
            "tanks": len(self.tanks()),
            "decouplers": len(self.decouplers()),
            "pods": len(self.pods()),
            "parachutes": len(self.parachutes()),
            "science": len(self.science()),
        }


# ==========================================================================
def _part_from_node(node: ConfigNode, cfg_path: Path) -> PartInfo | None:
    name = node.get("name")
    if not name:
        return None
    category = node.get("category", "")
    if category.lower() == "none":
        return None
    if str(node.get("TechHidden", "False")).lower() == "true":
        return None
    if "Prebuilt" in str(cfg_path):
        return None

    rescale = node.get_float("rescaleFactor", 1.0)
    info = PartInfo(
        name=name,
        title=localized(node, "title", name),
        category=category,
        cfg_path=str(cfg_path),
        dry_mass=node.get_float("mass", 0.0),
        cost=node.get_float("cost", 0.0),
        crash_tolerance=node.get_float("crashTolerance", 6.0),
        rescale=rescale,
        crew_capacity=node.get_int("CrewCapacity", 0),
        tags=localized(node, "tags", ""),
        bulkhead_profiles=node.get("bulkheadProfiles", ""),
    )

    # стыковочные ноды
    for key, raw in node.values:
        if not key.startswith("node_"):
            continue
        nums = []
        for chunk in raw.replace(",", " ").split():
            try:
                nums.append(float(chunk))
            except ValueError:
                break
        if len(nums) < 3:
            continue
        pos = (nums[0] * rescale, nums[1] * rescale, nums[2] * rescale)
        orient = tuple(nums[3:6]) if len(nums) >= 6 else (0.0, 1.0, 0.0)
        size = int(nums[6]) if len(nums) >= 7 else 1
        if key.startswith("node_stack_"):
            info.stack_nodes.append(AttachNode(key[len("node_stack_"):], pos, orient, size))
        elif key == "node_attach":
            info.surface_node = pos
            info.surface_node_orient = orient

    rules = node.get("attachRules", "")
    if rules:
        flags = [f.strip() for f in rules.split(",")]
        if len(flags) >= 2:
            info.can_surface_attach = flags[1] == "1"

    # ресурсы
    for res in node.nodes("RESOURCE"):
        rname = res.get("name")
        if rname:
            info.resources[rname] = res.get_float("maxAmount", res.get_float("amount", 0.0))

    # модули
    for mod in node.nodes("MODULE"):
        mname = mod.get("name", "")
        if not mname:
            continue
        info.modules.append(mname)
        if mname in ENGINE_MODULES:
            thrust = mod.get_float("maxThrust", 0.0)
            if thrust > info.max_thrust:
                info.max_thrust = thrust
                curve = mod.node("atmosphereCurve")
                if curve is not None:
                    for raw in curve.get_all("key"):
                        parts_ = raw.split()
                        if len(parts_) >= 2:
                            try:
                                p, isp = float(parts_[0]), float(parts_[1])
                            except ValueError:
                                continue
                            if abs(p) < 1e-9:
                                info.isp_vac = isp
                            elif abs(p - 1.0) < 1e-9:
                                info.isp_asl = isp
                info.propellants = [pr.get("name", "") for pr in mod.nodes("PROPELLANT")]
        elif mname == "ModuleGimbal":
            info.gimbal = max(info.gimbal, mod.get_float("gimbalRange", 0.0))
        elif mname in SOLAR_MODULES:
            # СЛЕДИТ ЛИ ПАНЕЛЬ ЗА СОЛНЦЕМ.
            #
            # Модуль у неподвижной накладки и у поворотной панели один и
            # тот же, различает их только это поле. Без него выбиралась
            # самая лёгкая панель — неподвижная OX-STAT, а она даёт ток,
            # лишь если случайно смотрит на солнце. Замер живого полёта:
            # обе панели раскрыты, `поток = 0.00`, и заряд с 110 ЕЭ упал
            # до нуля — зонд обесточился и перестал слушаться.
            # Поле зовётся `isTracking`, и в файлах поворотных панелей его
            # ПРОСТО НЕТ — у игры оно включено по умолчанию. Неподвижная
            # накладка задаёт его явно:
            #
            #     radialFlatSolarPanel.cfg:  isTracking = false
            #     1x6SolarPanels.cfg:        (поля нет)
            #
            # Поэтому умолчание здесь «истина», а не «ложь».
            info.sun_tracking = (mod.get("isTracking", "true").strip().lower()
                                 != "false")
            info.charge_rate = max(info.charge_rate,
                                   mod.get_float("chargeRate", 0.0))
        elif mname == "ModuleReactionWheel":
            info.roll_torque = max(info.roll_torque,
                                   mod.get_float("RollTorque", 0.0))
        elif mname in ANTENNA_MODULES:
            info.antenna_power = max(info.antenna_power,
                                     mod.get_float("antennaPower", 0.0))
            if mod.get("antennaType", "").strip().upper() == "RELAY":
                info.is_relay = True

    if info.is_engine and info.isp_asl <= 0:
        info.isp_asl = info.isp_vac * 0.8

    # МАСШТАБ УЗЛА ПОВЕРХНОСТНОГО КРЕПЛЕНИЯ.
    #
    # `node_attach` записан в МОДЕЛЬНЫХ единицах и умножается игрой на
    # scale × rescaleFactor. У большинства деталей оба равны единице, и
    # разницы не видно. А у винглетов scale = 0.1:
    #
    #     AV-R8: узел 4.324 -> 4.324 x 0.1 x 1.25 = 0.541 м
    #
    # Без масштаба сборщик выносил оперение на 4.3 метра от борта. В игре
    # это выглядело как ножки, растопыренные шире стартового стола, и
    # ракета заваливалась ещё до зажигания — снимок из игры это и показал.
    #
    # Умножается ТОЛЬКО узел поверхностного крепления. Стековые узлы здесь
    # не трогаются намеренно: по ним считаются высоты и диаметры баков,
    # они давно работают, а умолчание rescaleFactor = 1.25 задело бы все
    # детали, которые его не указывают.
    if info.surface_node is not None:
        scale = node.get_float("scale", 1.0) or 1.0
        # Умолчание KSP для деталей, где rescaleFactor не указан
        rescale = node.get_float("rescaleFactor", 1.25) or 1.25
        factor = scale * rescale
        if abs(factor - 1.0) > 1e-9:
            info.surface_node = tuple(v * factor for v in info.surface_node)
        info.rescale = factor
    return info


def _find_gamedata() -> Path | None:
    from ..agent.provision import find_ksp_root
    root = find_ksp_root()
    if root is None:
        return None
    gamedata = Path(root) / "GameData"
    return gamedata if gamedata.exists() else None


def _signature(gamedata: Path) -> str:
    """Отпечаток состава GameData — для инвалидации кэша."""
    newest = 0.0
    count = 0
    for cfg in gamedata.rglob("*.cfg"):
        try:
            newest = max(newest, cfg.stat().st_mtime)
            count += 1
        except OSError:
            continue
    return f"{gamedata}|{count}|{int(newest)}"


_CATALOG: PartCatalog | None = None


def get_catalog(refresh: bool = False) -> PartCatalog:
    """Глобальный каталог (ленивая загрузка + кэш)."""
    global _CATALOG
    if _CATALOG is None or refresh:
        root = CONFIG.craft.ksp_root
        gamedata = Path(root) / "GameData" if root else None
        if gamedata is not None and not gamedata.exists():
            gamedata = None
        _CATALOG = PartCatalog.load(gamedata, use_cache=not refresh)
    return _CATALOG
