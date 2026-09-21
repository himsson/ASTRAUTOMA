"""Rocket design for a mission target — with real KSP parts.

The user picks the layout (number of stages, side boosters, crew, payload),
and the designer returns a concrete parts list from the game's catalog:

    per stage: engine × count, fuel tanks × count, decoupler, fuel type,
               Δv, TWR, mass;
    required equipment: command, reaction wheel, batteries, solar
               panels, antenna, landing legs, fairing — by in-game name;
    optional equipment: parachutes, science, RCS, docking port.

Every stage is sized with the rocket equation using the REAL masses of
the chosen engine and tanks, then the Δv is recomputed from the parts that
were actually picked — so the numbers on screen are what the stage gives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import mission as ms
from .analysis import load_genome
from .craft import G0, catalog
from .i18n import L
from .world import World

UNIT_MASS = 0.005                       # t per unit of LF or Ox
SIZE_OF_PROFILE = {"size0": 0.625, "size1": 1.25, "size1p5": 1.875, "size2": 2.5,
                   "size3": 3.75, "size4": 5.0}
# The catalog's attach-node sizes are unreliable for some stock engines
ENGINE_SIZE = {
    "microEngine_v2": 0.625, "liquidEngineMini_v2": 0.625,
    "liquidEngine3_v2": 1.25, "liquidEngine_v2": 1.25, "liquidEngine2_v2": 1.25,
    "toroidalAerospike": 1.25, "nuclearEngine": 1.25, "LiquidEngineLV-T91": 1.25,
    "LiquidEngineRK-7": 1.25, "SSME": 1.25,
    "LiquidEngineLV-TX87": 1.875, "LiquidEngineRE-I2": 1.875, "LiquidEngineRE-J10": 1.875,
    "LiquidEngineKE-1": 1.875,
    "liquidEngine2-2_v2": 2.5, "engineLargeSkipper_v2": 2.5, "liquidEngineMainsail_v2": 2.5,
    "Size2LFB_v2": 2.5,
    "Size3AdvancedEngine": 3.75, "Size3EngineCluster": 3.75,
}
EXCLUDED_ENGINES = {"RAPIER", "ionEngine", "LaunchEscapeSystem", "sepMotor1", "Size1p5_Tank_05",
                    "LiquidEngineRV-1"}      # verniers are not main engines
DECOUPLER = {0.625: "Decoupler_0", 1.25: "Decoupler_1", 1.875: "Decoupler_1p5", 2.5: "Decoupler_2",
             3.75: "Decoupler_3", 5.0: "Decoupler_4"}
ENGINE_PLATE = {1.25: "EnginePlate5", 1.875: "EnginePlate1p5", 2.5: "EnginePlate2",
                3.75: "EnginePlate3", 5.0: "EnginePlate4"}
FAIRING = {1.25: "fairingSize1", 1.875: "fairingSize1p5", 2.5: "fairingSize2", 3.75: "fairingSize3",
           5.0: "fairingSize4"}
SIZES = (0.625, 1.25, 1.875, 2.5, 3.75, 5.0)
CLUSTERS = (1, 2, 3, 4, 5, 7, 9)


def title(name: str) -> str:
    p = catalog().parts.get(name)
    if p is None:
        return name
    return p.title.split("//")[0].replace("\\u0020", "").strip()


def part(name: str):
    return catalog().parts.get(name)


# ==========================================================================
@dataclass
class DesignConfig:
    stages: int = 0          # 0 — auto by target
    boosters: int = 0        # 0 / 2 / 4 / 6 / 8 side solid boosters
    crew: int = 0            # 0 — probe, 1 or 3 kerbals
    payload: float = 1.0     # t, extra payload on top of the equipment
    nuclear: bool = False    # allow LV-N on vacuum stages
    relay: bool = False      # design a comms relay satellite for the target

    def stage_count(self, target: ms.Target, total_dv: float = 0.0) -> int:
        """Auto: more Δv — more stages (each stage carries ~3 km/s at most)."""
        if self.stages:
            return self.stages
        if total_dv > 8_500:
            return 4
        if getattr(target, "planet", False) or target.code in ("LLO", "HLO", "LND"):
            return 3
        return 2


@dataclass
class Item:
    name: str               # part name in cfg
    count: int = 1

    @property
    def title(self) -> str:
        return title(self.name)


@dataclass
class StageDesign:
    label: str
    role: str
    dv_need: float
    twr_need: float
    g: float
    atmospheric: bool
    engine: Item | None = None
    tanks: list[Item] = field(default_factory=list)
    extras: list[Item] = field(default_factory=list)
    fuel_type: str = ""
    propellant: float = 0.0     # t
    units: float = 0.0
    m0: float = 0.0
    m1: float = 0.0
    thrust: float = 0.0         # kN (sea level for atmospheric stages)
    isp: float = 0.0
    dv: float = 0.0
    size: float = 1.25
    feasible: bool = True
    note: str = ""

    @property
    def twr(self) -> float:
        return self.thrust / (self.m0 * self.g) if self.m0 else 0.0


@dataclass
class Design:
    target: ms.Target
    config: DesignConfig
    budget: ms.Budget
    stages: list[StageDesign] = field(default_factory=list)      # bottom first
    required: list[tuple[str, Item, str]] = field(default_factory=list)
    optional: list[tuple[str, Item, str]] = field(default_factory=list)
    equipment_mass: float = 0.0
    top_size: float = 1.25

    @property
    def total_mass(self) -> float:
        return self.stages[0].m0 if self.stages and self.stages[0].m0 else 0.0

    @property
    def total_dv(self) -> float:
        return sum(s.dv for s in self.stages)

    @property
    def feasible(self) -> bool:
        return bool(self.stages) and all(s.feasible for s in self.stages)


# ==========================================================================
# Parts
# ==========================================================================
def _size(p) -> float:
    if p.name in ENGINE_SIZE:
        return ENGINE_SIZE[p.name]
    for prof in (p.bulkhead_profiles or "").replace(" ", "").split(","):
        if prof in SIZE_OF_PROFILE:
            return SIZE_OF_PROFILE[prof]
    return 0.0


def _engines(vacuum: bool, nuclear: bool):
    out = []
    for p in catalog().parts.values():
        if not p.is_engine or p.name in EXCLUDED_ENGINES or p.is_solid_booster:
            continue
        props = set(p.propellants)
        lfox = props >= {"LiquidFuel", "Oxidizer"} and "IntakeAir" not in props
        nerv = props == {"LiquidFuel"} and p.isp_vac > 600
        if not (lfox or (nerv and nuclear and vacuum)):
            continue
        if _size(p) <= 0:
            continue                       # radial-only engines are not stack engines
        if not vacuum and p.isp_asl < 200:
            continue                       # vacuum nozzles do not lift off
        out.append(p)
    return out


def _tanks(size: float, lf_only: bool = False):
    out = []
    for p in catalog().parts.values():
        if p.is_engine:
            continue
        lf = p.resources.get("LiquidFuel", 0)
        ox = p.resources.get("Oxidizer", 0)
        if lf_only:
            if not (lf > 0 and ox == 0):
                continue
        elif not (lf > 0 and ox > 0):
            continue
        profs = [x for x in (p.bulkhead_profiles or "").replace(" ", "").split(",") if x != "srf"]
        if "toroid" in p.name.lower():
            continue                       # radial doughnut tanks do not stack
        if len(profs) != 1 or SIZE_OF_PROFILE.get(profs[0]) != size:
            continue                       # adapters and fuselages are not plain tanks
        out.append((p, lf + ox))
    return sorted(out, key=lambda x: -x[1])


def _fill_tanks(family, units_needed: float) -> list[tuple]:
    """Fewest tanks that hold at least the needed units (largest first)."""
    counts: dict[str, list] = {}
    left = units_needed
    for p, u in family:
        n = int(left // u)
        if n:
            counts.setdefault(p.name, [p, 0])[1] += n
            left -= n * u
    if left > 0:
        # the smallest tank that covers the rest
        fit = [x for x in family if x[1] >= left]
        p, u = (fit[-1] if fit else family[0])
        counts.setdefault(p.name, [p, 0])[1] += 1
    return [(p, n) for p, n in counts.values()]


def _rocket(payload: float, dv: float, isp: float, dry_fixed: float, tank_ratio: float) -> float | None:
    """Wet mass of a stage: payload + fixed dry + propellant with tank dry mass."""
    f = 1.0 - math.exp(-dv / (isp * G0))
    denom = 1.0 - f * (1.0 + tank_ratio)
    if denom <= 0.05:
        return None
    return (payload + dry_fixed) / denom


def _size_stage(st: StageDesign, payload: float, nuclear: bool, min_size: float) -> None:
    vacuum = not st.atmospheric
    best = None
    for eng in _engines(vacuum, nuclear):
        e_size = _size(eng)
        isp = eng.isp_vac if vacuum else eng.isp_asl * 0.45 + eng.isp_vac * 0.55
        thrust_one = eng.max_thrust if vacuum else eng.thrust_at(1.0)
        nerv = "Oxidizer" not in eng.propellants
        for n in CLUSTERS:
            size = e_size if n == 1 else next((s for s in SIZES if s > e_size * 1.4), 5.0)
            size = max(size, min_size)
            if n > 1 and size not in ENGINE_PLATE:
                continue
            family = _tanks(size, lf_only=nerv)
            if not family:
                continue
            ratio = sum(p.dry_mass for p, _ in family) / sum(u * UNIT_MASS for _, u in family)
            dry = eng.dry_mass * n + part(DECOUPLER[size]).dry_mass
            if n > 1:
                dry += part(ENGINE_PLATE[size]).dry_mass
            m0 = _rocket(payload, st.dv_need, isp, dry, ratio)
            if m0 is None or n * thrust_one < st.twr_need * m0 * st.g:
                continue
            score = m0 * (1 + 0.02 * (n - 1))
            if best is None or score < best[0]:
                best = (score, eng, n, size, family, isp, dry, m0, nerv)
    if best is None:
        st.feasible = False
        return
    _, eng, n, size, family, isp, dry, m0, nerv = best
    f = 1.0 - math.exp(-st.dv_need / (isp * G0))
    tanks = _fill_tanks(family, m0 * f / UNIT_MASS)
    real_units = sum(p.fuel_units() * k for p, k in tanks)
    tank_dry = sum(p.dry_mass * k for p, k in tanks)
    st.engine = Item(eng.name, n)
    st.tanks = [Item(p.name, k) for p, k in tanks]
    st.extras = [Item(DECOUPLER[size])]
    if n > 1:
        st.extras.append(Item(ENGINE_PLATE[size]))
    st.fuel_type = (L("Liquid Fuel only (nuclear)", "только жидкое топливо (ядерный)") if nerv
                    else L("Liquid Fuel + Oxidizer", "жидкое топливо + окислитель"))
    st.units = real_units
    st.propellant = real_units * UNIT_MASS
    st.m1 = payload + dry + tank_dry
    st.m0 = st.m1 + st.propellant
    st.isp = isp
    st.dv = isp * G0 * math.log(st.m0 / st.m1)
    st.thrust = n * (eng.max_thrust if vacuum else eng.thrust_at(1.0))
    st.size = size


def _boosters(count: int, core_m0: float, dv_share: float, g: float, core_thrust: float,
              liftoff_twr: float):
    """Side solid boosters: the smallest SRB that carries its share of the climb."""
    srbs = [p for p in catalog().parts.values()
            if p.is_solid_booster and p.name not in EXCLUDED_ENGINES and p.max_thrust >= 100]
    best = None
    for srb in sorted(srbs, key=lambda p: p.dry_mass):
        sf = srb.resources.get("SolidFuel", 0.0) * 0.0075
        wet = srb.dry_mass + sf
        isp = srb.isp_asl * 0.5 + srb.isp_vac * 0.5
        m0 = core_m0 + count * (wet + part("radialDecoupler").dry_mass)
        m1 = m0 - count * sf
        dv = isp * G0 * math.log(m0 / m1)
        thrust = count * srb.thrust_at(1.0) + core_thrust
        if thrust < liftoff_twr * m0 * g:
            continue
        best = (srb, m0, m1, dv, thrust, isp)
        if dv >= dv_share:
            break
    return best


# ==========================================================================
def _stage_plan(budget: ms.Budget, n: int, factor: float,
                split: float) -> list[tuple[str, float, bool]]:
    """(role, Δv, atmospheric) for each stage, bottom first."""
    legs = {l.key: l.dv * factor for l in budget.legs}
    asc = legs.get("ascent", 0.0)
    after = sum(v for k, v in legs.items() if k not in ("ascent", "loi", "land"))
    lander = legs.get("loi", 0.0) + legs.get("land", 0.0)
    to_orbit = L("ascent", "выведение")
    if n == 1:
        return [(L("everything", "всё"), sum(legs.values()), True)]
    top = []
    if lander and n >= 3:
        top = [(L("capture + landing", "захват + посадка") if legs.get("land")
                else L("capture at the Moon", "захват у Луны"), lander, False)]
        n_lower, rest_after = n - 1, after
    else:
        n_lower, rest_after = n, after + lander
    if n_lower == 1:
        return [(to_orbit, asc + rest_after, True)] + top
    shares = {2: (split, 1 - split), 3: (0.40, 0.40, 0.20), 4: (0.33, 0.33, 0.22, 0.12)}
    parts = shares.get(n_lower, tuple([1 / n_lower] * n_lower))
    plan = []
    for i, share in enumerate(parts):
        dv = asc * share
        role = to_orbit
        if i == len(parts) - 1 and rest_after:
            dv += rest_after
            role = L("orbit + transfer", "орбита + перелёт")
        plan.append((role, dv, i == 0))
    return plan + top


def design(world: World, target: ms.Target, cfg: DesignConfig, margin: bool = True) -> Design:
    if cfg.relay:
        from .relays import relay_target
        target = relay_target(world, target, 0, 1, 2)
    budget = ms.budget(world, target, gear=True)
    d = Design(target, cfg, budget)
    h = world.home_body
    genome = load_genome().get("genome", {}).get("design", {})
    split = float(genome.get("ascent_split", 0.55))
    liftoff = max(1.3, float(genome.get("liftoff_twr", 1.5)))
    upper_twr = max(0.6, float(genome.get("upper_twr", 0.75)))
    factor = ms.MARGIN if margin else 1.0

    _equipment(d, world)
    n = cfg.stage_count(target, budget.total_margin)
    plan = _stage_plan(budget, n, factor, split)
    labels = ["I", "II", "III", "IV", "V"]
    stages = []
    for i, (role, dv, atm) in enumerate(plan):
        is_lander = target.landing and i == len(plan) - 1 and n >= 2
        twr = liftoff if i == 0 else (2.0 if is_lander else upper_twr)
        g = ms.target_body(world, target).g0 if is_lander else h.g0
        stages.append(StageDesign(L(f"Stage {labels[i]}", f"Ступень {labels[i]}"), role, dv, twr, g,
                                  atmospheric=atm))

    boost_share = 0.0
    if cfg.boosters:
        boost_share = stages[0].dv_need * 0.35
        stages[0].dv_need -= boost_share
        stages[0].twr_need = 0.7            # boosters carry the liftoff

    load = d.equipment_mass + cfg.payload
    min_size = d.top_size
    for st in reversed(stages):
        _size_stage(st, load, cfg.nuclear, min_size)
        if not st.feasible:
            st.note = L("not reachable — add a stage or boosters",
                        "недостижимо — добавьте ступень или ускорители")
            break
        load = st.m0
        min_size = max(min_size, st.size)

    if cfg.boosters and stages[0].feasible:
        res = _boosters(cfg.boosters, stages[0].m0, boost_share, h.g0, stages[0].thrust, liftoff)
        if res:
            srb, m0, m1, dv, thrust, isp = res
            b = StageDesign(L("Boosters", "Ускорители"), L("liftoff", "старт"), boost_share,
                            liftoff, h.g0, atmospheric=True)
            b.engine = Item(srb.name, cfg.boosters)
            b.extras = [Item("radialDecoupler2" if srb.dry_mass > 2 else "radialDecoupler",
                             cfg.boosters)]
            b.fuel_type = L("Solid Fuel", "твёрдое топливо")
            b.units = srb.resources.get("SolidFuel", 0.0) * cfg.boosters
            b.propellant = b.units * 0.0075
            b.m0, b.m1, b.dv, b.thrust, b.isp = m0, m1, dv, thrust, isp
            if dv < boost_share * 0.8:
                b.note = L("the boosters give less than planned — stage I carries the rest",
                           "ускорители дают меньше плана — ступень I доберёт остальное")
            stages.insert(0, b)
        else:
            stages[0].note = L("no booster lifts this rocket — try more boosters",
                               "ни один ускоритель не поднимет ракету — поставьте больше")
            stages[0].feasible = False
    d.stages = stages
    return d


# ==========================================================================
def _equipment(d: Design, world: World) -> None:
    """Required and optional equipment by in-game part name."""
    cfg, target = d.config, d.target
    h = world.home_body
    st = world.settings
    req, opt = d.required, d.optional

    if cfg.crew >= 3:
        pod, size = "mk1-3pod", 2.5
    elif cfg.crew >= 1:
        pod, size = "mk1pod_v2", 1.25
    else:
        pod, size = ("probeStackSmall", 1.25) if cfg.payload < 8 else ("probeStackLarge", 2.5)
    d.top_size = size
    req.append((L("Command", "Управление"), Item(pod), L("controls the craft", "управляет аппаратом")))
    wheel = "sasModule" if size <= 1.25 else "asasmodule1-2"
    req.append((L("Reaction wheel", "Маховик"), Item(wheel),
                L("turns the craft without the engine", "поворачивает аппарат без двигателя")))
    mass = part(pod).dry_mass + part(wheel).dry_mass

    far = target.body != h.name
    tb = ms.target_body(world, target)
    rate = 0.04 + 0.01 * cfg.crew
    night = (tb.period_at(ms.work_altitude(world, target) or 100_000.0) / 2 if far
             else h.period_at(ms.leo_altitude(h)) / 2)
    need_ec = max(rate * night * 1.5, 200.0)
    pod_ec = part(pod).resources.get("ElectricCharge", 0)
    batt = "batteryBank" if need_ec > 600 else "ksp_r_largeBatteryPack"
    n_batt = max(0, math.ceil((need_ec - pod_ec) / part(batt).resources.get("ElectricCharge", 400)))
    if n_batt:
        req.append((L("Batteries", "Аккумуляторы"), Item(batt, n_batt),
                    L(f"≥ {need_ec:,.0f} EC for the night side",
                      f"≥ {need_ec:,.0f} EC на теневую сторону").replace(",", " ")))
        mass += part(batt).dry_mass * n_batt
    panel = "solarPanels4"
    n_panels = max(2, math.ceil(rate * 3 / part(panel).charge_rate))
    n_panels += n_panels % 2
    req.append((L("Solar panels", "Солнечные панели"), Item(panel, n_panels),
                L("deploy after the fairing, symmetric", "раскрыть после обтекателя, по симметрии")))
    mass += part(panel).dry_mass * n_panels

    if cfg.relay:
        # A relay: the strongest link it needs back home, lots of power, no legs
        distance = ms.comm_distance(world, target)
        need_power = distance ** 2 / st.dsn_power / max(st.range_modifier, 1e-6)
        relays = sorted([p for p in catalog().parts.values() if p.is_relay and p.antenna_power > 0],
                        key=lambda p: p.antenna_power)
        if relays:
            pick = next((p for p in relays if p.antenna_power >= need_power * 1.3), relays[-1])
            req.append((L("Relay antenna", "Антенна-ретранслятор"), Item(pick.name),
                        L("links the craft at the target with home", "связывает аппарат у цели с домом")))
            mass += pick.dry_mass
        d.equipment_mass = mass
        opt.append((L("Science", "Наука"), Item("sensorThermometer"), ""))
        return
    if st.commnet:
        distance = ms.comm_distance(world, target)
        need_power = distance ** 2 / st.dsn_power / max(st.range_modifier, 1e-6)
        antennas = sorted([p for p in catalog().parts.values()
                           if p.is_antenna and p.antenna_power > 0 and not p.is_command],
                          key=lambda p: p.antenna_power)
        if need_power * 1.3 > 5000 and antennas:
            pick = next((p for p in antennas if p.antenna_power >= need_power * 1.3), antennas[-1])
            req.append((L("Antenna", "Антенна"), Item(pick.name),
                        L(f"range for {distance / 1e6:,.1f} Mm",
                          f"дальность на {distance / 1e6:,.1f} тыс. км").replace(",", " ")))
            mass += pick.dry_mass
        if far and st.require_signal:
            opt.append((L("Relay satellite", "Спутник-ретранслятор"), Item("RelayAntenna50"),
                        L("keeps control over the far side", "связь над обратной стороной")))

    if target.landing:
        lander_mass = mass + cfg.payload
        leg = ("miniLandingLeg" if lander_mass < 2
               else "landingLeg1" if lander_mass < 12 else "landingLeg1-2")
        n_legs = 3 if lander_mass < 2 else 4
        req.append((L("Landing legs", "Посадочные опоры"), Item(leg, n_legs),
                    L("symmetric, wide stance", "по симметрии, широкая база")))
        mass += part(leg).dry_mass * n_legs
    if far and tb.atmosphere_depth > 0:
        shield = {1.25: "HeatShield1", 1.875: "HeatShield1p5", 2.5: "HeatShield2",
                  3.75: "HeatShield3"}.get(size, "HeatShield1")
        req.append((L("Heat shield", "Теплозащитный экран"), Item(shield),
                    L(f"aerocapture and entry at {tb.name}", f"аэрозахват и вход в атмосферу {tb.name}")))
        mass += part(shield).dry_mass + part(shield).resources.get("Ablator", 0) * 0.001
        if target.landing:
            chute = "parachuteRadial" if size > 1.25 else "parachuteSingle"
            n_ch = 4 if tb.atmosphere_depth < 60_000 else 2      # thin air — more canopies
            req.append((L("Parachutes", "Парашюты"), Item(chute, n_ch),
                        L(f"slow down in the {tb.name} atmosphere", f"торможение в атмосфере {tb.name}")))
            mass += part(chute).dry_mass * n_ch
    if h.atmosphere_depth > 0:
        fairing = FAIRING.get(size, "fairingSize1")
        req.append((L("Fairing", "Обтекатель"), Item(fairing),
                    L("over the payload for the climb", "поверх нагрузки на выведение")))
        mass += part(fairing).dry_mass

    big = size > 1.25
    opt.append((L("Parachutes", "Парашюты"),
                Item("parachuteRadial" if big else "parachuteSingle", 3 if big else 1),
                L("to bring the capsule home", "вернуть капсулу домой")))
    opt.append((L("Science", "Наука"), Item("sensorThermometer"),
                L("cheap experiment, run by the pilot", "дешёвый эксперимент, пилот запускает сам")))
    opt.append((L("Science", "Наука"), Item("GooExperiment"), ""))
    opt.append(("RCS", Item("RCSBlock_v2", 4),
                L("+ a Stratus-V tank, for precise docking", "+ бак Stratus-V, для точной стыковки")))
    if cfg.crew == 0 and target.landing:
        opt.append((L("Crewed lander", "Пилотируемая посадка"), Item("landerCabinSmall"),
                    L("instead of the probe core, for a kerbal on the surface",
                      "вместо ядра зонда, если нужен кербонавт на поверхности")))
    opt.append((L("Docking port", "Стыковочный узел"), Item("dockingPort2"),
                L("for assembly in orbit", "для сборки на орбите")))
    d.equipment_mass = mass
