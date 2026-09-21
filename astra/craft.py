"""Загруженный чертёж: из чего собрана ракета и сколько она может.

Два источника, одна модель:
  * игра подключена и идёт полёт — читается активный аппарат через kRPC
    (реальные остатки топлива, реальная схема ступеней);
  * иначе — самый свежий .craft сохранения (то, что последним сохранено
    или запущено из VAB/SPH), параметры деталей — из каталога GameData.

По модели считается Δv каждой ступени так же, как это делает игра:
ступени срабатывают от старшей к нулевой, при срабатывании ступени
отпадает всё, что висит под её разделителями.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

G0 = 9.80665
ROOT = Path(__file__).resolve().parent.parent
PROPELLANTS_LIQUID = ("LiquidFuel", "Oxidizer")
DENSITY = {"LiquidFuel": 0.005, "Oxidizer": 0.005, "SolidFuel": 0.0075,
           "MonoPropellant": 0.004, "XenonGas": 0.0001, "Ore": 0.010,
           "Ablator": 0.001, "ElectricCharge": 0.0}


# ==========================================================================
@dataclass
class PartModel:
    name: str
    title: str
    dry: float                                   # т
    resources: dict[str, float] = field(default_factory=dict)       # текущее
    resources_max: dict[str, float] = field(default_factory=dict)
    activate_stage: int = -1
    decouple_stage: int = -1
    # двигатель
    thrust_vac: float = 0.0                      # кН
    isp_vac: float = 0.0
    isp_asl: float = 0.0
    solid: bool = False
    propellants: tuple = ("LiquidFuel", "Oxidizer")
    # оборудование
    command: bool = False
    crew_capacity: int = 0
    crew: int = 0
    reaction_wheel: bool = False
    rcs: bool = False
    antenna_power: float = 0.0
    relay: bool = False
    solar_rate: float = 0.0                      # EC/с при полном солнце
    generator_rate: float = 0.0                  # РИТЭГ, топливные элементы
    leg: bool = False
    parachute: bool = False
    decoupler: bool = False
    fairing: bool = False
    gimbal: float = 0.0

    @property
    def is_engine(self) -> bool:
        return self.thrust_vac > 0 and self.isp_vac > 0

    @property
    def wet(self) -> float:
        return self.dry + sum(DENSITY.get(r, 0.0) * a for r, a in self.resources.items())

    def isp_at(self, p_atm: float) -> float:
        p = max(0.0, min(1.0, p_atm))
        return self.isp_vac + (self.isp_asl - self.isp_vac) * p

    def thrust_at(self, p_atm: float) -> float:
        return self.thrust_vac * self.isp_at(p_atm) / self.isp_vac if self.isp_vac else 0.0


@dataclass
class StageResult:
    stage: int
    dv_vac: float
    dv_asl: float
    m0: float
    m1: float
    thrust_vac: float
    thrust_asl: float
    isp_vac: float
    isp_asl: float
    burn_time: float
    propellant_units: float

    def twr(self, g: float, atm: bool = False) -> float:
        thrust = self.thrust_asl if atm else self.thrust_vac
        return thrust / (self.m0 * g) if self.m0 else 0.0

    def dv_mixed(self, share_atm: float) -> float:
        return self.dv_asl * share_atm + self.dv_vac * (1.0 - share_atm)


@dataclass
class Vessel:
    name: str
    source: str                    # «активный аппарат» / путь к .craft
    parts: list[PartModel]
    situation: str = "pre_launch"
    modified: float = 0.0
    unknown_parts: list[str] = field(default_factory=list)

    # --- сводки ---------------------------------------------------------
    @property
    def mass(self) -> float:
        return sum(p.wet for p in self.parts)

    def count(self, attr: str) -> int:
        return sum(1 for p in self.parts if getattr(p, attr))

    @property
    def crew(self) -> int:
        return sum(p.crew for p in self.parts)

    @property
    def crewed(self) -> bool:
        return self.crew > 0

    @property
    def ec_capacity(self) -> float:
        return sum(p.resources_max.get("ElectricCharge", 0.0) for p in self.parts)

    @property
    def solar_rate(self) -> float:
        return sum(p.solar_rate for p in self.parts)

    @property
    def generator_rate(self) -> float:
        return sum(p.generator_rate for p in self.parts)

    @property
    def best_antenna(self) -> float:
        """Суммарная мощность антенн по правилу CommNet: сильнейшая плюс
        вклад остальных, объединяемых с ней (упрощённо — сумма)."""
        powers = sorted((p.antenna_power for p in self.parts if p.antenna_power > 0),
                        reverse=True)
        if not powers:
            return 0.0
        strongest = powers[0]
        rest = sum(powers[1:])
        return strongest + rest * 0.75 if rest else strongest

    @property
    def liquid_units(self) -> float:
        return sum(p.resources.get(r, 0.0) for p in self.parts for r in PROPELLANTS_LIQUID)

    @property
    def consumption(self) -> float:
        """Оценка постоянного потребления EC/с: ядра зондов и приборы."""
        rate = 0.0
        for p in self.parts:
            if p.command and p.crew_capacity == 0:
                rate += 0.025
            elif p.command:
                rate += 0.01
        return rate

    # --- ступени ----------------------------------------------------------
    def stages(self) -> list[StageResult]:
        parts = [PartModel(**{**p.__dict__, "resources": dict(p.resources)})
                 for p in self.parts]
        if not parts:
            return []
        top = max([p.activate_stage for p in parts] + [p.decouple_stage + 1 for p in parts] + [0])
        results = []
        for s in range(top, -1, -1):
            present = [p for p in parts if p.decouple_stage < s]
            engines = [p for p in present if p.is_engine and p.activate_stage >= s]
            if not engines:
                continue
            liquid = [e for e in engines if not e.solid]
            solid = [e for e in engines if e.solid]
            props = sorted({r for e in liquid for r in e.propellants})

            # Разделитель перекрывает подачу: двигатель не дотянется до
            # баков, которые останутся на ракете после его отстрела.
            reach = min((e.decouple_stage for e in liquid), default=-1)

            def has_fuel(p):
                return (p.decouple_stage >= reach
                        and any(p.resources.get(r, 0) > 0 for r in props))

            # Питает ступень бак, отпадающий первым из оставшихся.
            groups = sorted({p.decouple_stage for p in present}, reverse=True)
            pool = []
            for d in groups:
                pool = [p for p in present if p.decouple_stage == d and has_fuel(p)]
                if pool:
                    break
            m0 = sum(p.wet for p in present)
            burned = 0.0
            units = 0.0
            if liquid:
                for p in pool:
                    for r in props:
                        amt = p.resources.get(r, 0.0)
                        burned += amt * DENSITY.get(r, 0.005)
                        units += amt
                        p.resources[r] = 0.0
            for e in solid:
                amt = e.resources.get("SolidFuel", 0.0)
                burned += amt * DENSITY["SolidFuel"]
                e.resources["SolidFuel"] = 0.0
            if burned <= 0:
                continue
            m1 = m0 - burned
            tv = sum(e.thrust_vac for e in engines)
            ta = sum(e.thrust_at(1.0) for e in engines)
            isp_v = tv / sum(e.thrust_vac / e.isp_vac for e in engines)
            isp_a = ta / sum(e.thrust_at(1.0) / max(e.isp_asl, 1e-3) for e in engines) \
                if ta > 0 else 0.0
            dv_v = isp_v * G0 * math.log(m0 / m1)
            dv_a = isp_a * G0 * math.log(m0 / m1) if isp_a else 0.0
            flow = tv / (isp_v * G0)            # т/с
            results.append(StageResult(s, dv_v, dv_a, m0, m1, tv, ta, isp_v, isp_a,
                                       burned / flow if flow else 0.0, units))
        return results

    @property
    def dv_vac_total(self) -> float:
        return sum(s.dv_vac for s in self.stages())


# ==========================================================================
# Каталог деталей
# ==========================================================================
_CATALOG = None


def catalog():
    """Каталог деталей: из GameData, а без неё — из сохранённого кэша."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    from kia_core.engineer.parts_catalog import CACHE_FILE, PartCatalog, PartInfo, _find_gamedata
    gd = None
    try:
        gd = _find_gamedata()
    except Exception:
        gd = None
    if gd is not None:
        _CATALOG = PartCatalog.load(gd)
    elif CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        parts = {n: PartInfo.from_dict(d) for n, d in data["parts"].items()}
        _CATALOG = PartCatalog(parts, data.get("densities"), source="кэш")
    else:
        _CATALOG = PartCatalog.fallback()
    return _CATALOG


def _lookup(name: str):
    """Деталь по имени из .craft; старые имена (до 1.10) ищутся как *_v2."""
    parts = catalog().parts
    base = name.replace(".", "_")
    stem = re.sub(r"_v\d+$", "", base)
    for key in (base, name, f"{base}_v2", f"{stem}_v2", f"{stem}_v3", stem):
        if key in parts:
            return parts[key]
    return None


def _from_info(name: str, info) -> PartModel:
    mods = set(info.modules)
    lowered = f"{info.name} {info.title}".lower()
    gen = 0.0
    if "ModuleGenerator" in mods and not info.is_solar:
        gen = 0.75 if "rtg" in lowered else 0.0
    return PartModel(
        name=name, title=info.title, dry=info.dry_mass,
        resources=dict(info.resources), resources_max=dict(info.resources),
        thrust_vac=info.max_thrust if info.is_engine else 0.0,
        isp_vac=info.isp_vac, isp_asl=info.isp_asl,
        solid=info.is_solid_booster,
        propellants=tuple(r for r in info.propellants if r != "ElectricCharge")
        or ("LiquidFuel", "Oxidizer"),
        command=info.is_command, crew_capacity=info.crew_capacity,
        reaction_wheel="ModuleReactionWheel" in mods,
        rcs=bool({"ModuleRCS", "ModuleRCSFX"} & mods),
        antenna_power=info.antenna_power if info.is_antenna else 0.0,
        relay=info.is_relay,
        solar_rate=info.charge_rate if info.is_solar else 0.0,
        generator_rate=gen,
        leg=("ModuleWheelBase" in mods and "wheel" not in lowered) or "landingleg" in lowered,
        parachute=info.is_parachute, decoupler=info.is_decoupler,
        fairing="fairing" in lowered, gimbal=info.gimbal)


_DECOUPLE_CACHE: dict[str, tuple[str, bool]] = {}


def decouple_mode(info) -> tuple[str, bool]:
    """С какой стороны рвётся связь у разделителя: (узел, omni).

    Обычный разделитель рвётся по верхнему узлу и улетает вниз вместе с
    отработавшей ступенью. Пластина двигателя (EP-*) рвётся по НИЖНЕМУ
    узлу: отлетает то, что под ней, а сама она остаётся на ракете. Без
    этого различия двигатель второй ступени «отваливался» в момент
    собственного включения, и ступень давала 11 м/с вместо тысяч.
    """
    if info is None:
        return "top", False
    key = info.name
    if key in _DECOUPLE_CACHE:
        return _DECOUPLE_CACHE[key]
    mode = ("srf" if "ModuleAnchoredDecoupler" in info.modules else "top", False)
    try:
        from kia_core.engineer.config_node import parse_file
        root = parse_file(info.cfg_path)
        for part in root.walk() if hasattr(root, "walk") else []:
            if getattr(part, "name", "") != "PART" or part.get("name") != info.name:
                continue
            for mod in part.nodes("MODULE"):
                if mod.get("name") in ("ModuleDecouple", "ModuleAnchoredDecoupler"):
                    mode = ((mod.get("explosiveNodeID") or mode[0]).strip(),
                            (mod.get("isOmniDecoupler") or "").strip().lower() == "true")
                    break
            break
    except Exception:
        pass
    _DECOUPLE_CACHE[key] = mode
    return mode


# ==========================================================================
# Источник 1: файл .craft
# ==========================================================================
def parse_craft(path: Path) -> Vessel:
    text = path.read_text(encoding="utf-8", errors="ignore")
    ship = re.search(r"^\s*ship\s*=\s*(.+)$", text, re.M)
    blocks = re.findall(r"^PART\s*\{(.*?)^\}", text, re.S | re.M)
    raw: dict[str, dict] = {}
    order = []
    for block in blocks:
        pid = re.search(r"^\s*part\s*=\s*(.+)$", block, re.M)
        if not pid:
            continue
        key = pid.group(1).strip()
        order.append(key)
        istg = re.search(r"^\s*istg\s*=\s*(-?\d+)", block, re.M)
        links = re.findall(r"^\s*link\s*=\s*(.+)$", block, re.M)
        res = {}
        res_max = {}
        for rb in re.findall(r"RESOURCE\s*\{(.*?)\}", block, re.S):
            rn = re.search(r"name\s*=\s*(\S+)", rb)
            ra = re.search(r"\bamount\s*=\s*([\d.eE+-]+)", rb)
            rm = re.search(r"maxAmount\s*=\s*([\d.eE+-]+)", rb)
            if rn and ra:
                res[rn.group(1)] = float(ra.group(1))
                res_max[rn.group(1)] = float(rm.group(1)) if rm else float(ra.group(1))
        crew = len(re.findall(r"^\s*crew\s*=", block, re.M))
        attach = {}
        for node, other in re.findall(r"^\s*attN\s*=\s*([^,\s]+),(\S+?_\d+)(?:_|$)", block, re.M):
            attach[other] = node
        raw[key] = {"attach": attach, "istg": int(istg.group(1)) if istg else -1,
                    "links": [l.strip() for l in links], "res": res,
                    "res_max": res_max, "crew": crew}

    parent: dict[str, str] = {}
    for key, d in raw.items():
        for child in d["links"]:
            parent[child] = key

    vessel = Vessel(name=ship.group(1).strip() if ship else path.stem,
                    source=str(path), parts=[], modified=path.stat().st_mtime)
    models: dict[str, PartModel] = {}
    for key in order:
        name = key.rsplit("_", 1)[0]
        info = _lookup(name)
        if info is None:
            vessel.unknown_parts.append(name)
            m = PartModel(name=name, title=name, dry=0.05)
        else:
            m = _from_info(name, info)
        d = raw[key]
        if d["res"]:
            m.resources = dict(d["res"])
            m.resources_max = dict(d["res_max"])
        m.activate_stage = d["istg"] if (m.is_engine or m.decoupler or m.parachute
                                         or m.fairing) else -1
        m.crew = d["crew"]
        models[key] = m

    # Что отлетает при срабатывании разделителя: либо он сам со всем, что
    # висит под ним, либо (пластина двигателя) только деталь на его
    # взрывном узле, а сам он остаётся.
    sep_roots: dict[str, int] = {}
    for key, m in models.items():
        stage = raw[key]["istg"]
        if not m.decoupler or stage < 0:
            continue
        node, omni = decouple_mode(_lookup(m.name))
        root = key
        if not omni and node not in ("srf",):
            for child in raw[key]["links"]:
                if raw[key]["attach"].get(child) == node:
                    root = child
                    break
        sep_roots[root] = max(sep_roots.get(root, -1), stage)

    # Ступень отделения детали: самый ранний (старший) отстрел на пути к корню.
    for key, m in models.items():
        d_stage = -1
        cur = key
        while cur is not None:
            if cur in sep_roots:
                d_stage = max(d_stage, sep_roots[cur])
            cur = parent.get(cur)
        m.decouple_stage = d_stage
    vessel.parts = list(models.values())
    return vessel


def latest_craft(ksp_root: Path | None, save_name: str) -> Path | None:
    if ksp_root is None or not save_name:
        return None
    ships = ksp_root / "saves" / save_name / "Ships"
    files = [p for p in ships.glob("*/*.craft")] if ships.exists() else []
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


# ==========================================================================
# Источник 2: живой аппарат через kRPC
# ==========================================================================
def from_krpc(conn, progress=None) -> Vessel:
    """Активный аппарат из игры.

    ЕДИНИЦЫ: kRPC отдаёт массу в КИЛОГРАММАХ и тягу в НЬЮТОНАХ. Модель
    считает в тоннах и килоньютонах — без перевода масса выходила в
    тысячу раз больше, и анализ показывал Δv 9 м/с и TWR 0.01 у ракеты,
    которая на деле улетала.

    СКОРОСТЬ: каждый вопрос к детали — отдельный запрос к игре. Поэтому
    оборудование берётся готовыми списками (parts.engines, parts.legs…)
    и сопоставляется с деталями по идентификатору, а у самой детали
    спрашивается только необходимое.
    """
    v = conn.space_center.active_vessel
    all_parts = v.parts.all
    total = len(all_parts)

    def ids(items, attr="part"):
        out = {}
        for it in items:
            try:
                out[getattr(it, attr)._object_id] = it
            except Exception:
                pass
        return out

    P = v.parts
    lists = {}
    for key, getter in (("engine", lambda: P.engines), ("leg", lambda: P.legs),
                        ("antenna", lambda: P.antennas), ("solar", lambda: P.solar_panels),
                        ("wheel", lambda: P.reaction_wheels), ("rcs", lambda: P.rcs),
                        ("decoupler", lambda: P.decouplers), ("chute", lambda: P.parachutes),
                        ("fairing", lambda: P.fairings)):
        try:
            lists[key] = ids(getter())
        except Exception:
            lists[key] = {}
    try:
        lists["command"] = {p._object_id: p for p in P.with_module("ModuleCommand")}
    except Exception:
        lists["command"] = {}

    parts = []
    for n, p in enumerate(all_parts):
        if progress and n % 10 == 0:
            progress(n / max(total, 1))
        pid = p._object_id
        name = p.name
        info = _lookup(name)
        m = _from_info(name, info) if info is not None else PartModel(name=name, title=name, dry=0.0)
        m.dry = p.dry_mass / 1000.0
        m.activate_stage = p.stage
        m.decouple_stage = p.decouple_stage
        m.resources, m.resources_max = {}, {}
        try:
            res = p.resources
            for rname in res.names:
                m.resources[rname] = res.amount(rname)
                m.resources_max[rname] = res.max(rname)
        except Exception:
            pass
        eng = lists["engine"].get(pid)
        if eng is not None:
            try:
                m.thrust_vac = eng.max_vacuum_thrust / 1000.0
                m.isp_vac = eng.vacuum_specific_impulse
                m.isp_asl = eng.kerbin_sea_level_specific_impulse
                m.solid = bool(eng.throttle_locked)
                names = tuple(eng.propellant_names)
                if names:
                    m.propellants = tuple(r for r in names if r != "ElectricCharge")
            except Exception:
                pass
        else:
            m.thrust_vac = 0.0
        m.leg = pid in lists["leg"] or m.leg
        m.reaction_wheel = pid in lists["wheel"] or m.reaction_wheel
        m.rcs = pid in lists["rcs"] or m.rcs
        m.decoupler = pid in lists["decoupler"] or m.decoupler
        m.parachute = pid in lists["chute"] or m.parachute
        m.fairing = pid in lists["fairing"] or m.fairing
        m.command = pid in lists["command"] or m.command
        ant = lists["antenna"].get(pid)
        if ant is not None:
            try:
                m.antenna_power = ant.power
            except Exception:
                pass
        parts.append(m)
    try:
        crew = v.crew_count
        for m in parts:
            if m.command and m.crew_capacity:
                take = min(crew, m.crew_capacity)
                m.crew, crew = take, crew - take
    except Exception:
        pass
    if progress:
        progress(1.0)
    return Vessel(name=v.name, source="kRPC", parts=parts,
                  situation=str(v.situation).split(".")[-1], modified=0.0)


# ==========================================================================
def loaded(world, progress=None) -> Vessel | None:
    """Чертёж, который сейчас загружен."""
    conn = world.connection
    if conn is not None:
        try:
            if conn.in_flight():
                return from_krpc(conn, progress)
        except Exception:
            pass
    path = latest_craft(world.ksp_root, world.save_name)
    if path is None:
        return None
    try:
        return parse_craft(path)
    except Exception:
        return None


def describe_source(vessel: Vessel) -> str:
    if vessel.modified:
        when = datetime.fromtimestamp(vessel.modified).strftime("%d.%m %H:%M")
        return f"{Path(vessel.source).parent.name}/{Path(vessel.source).name} · {when}"
    return vessel.source
