"""Предполётный анализ: хватит ли загруженного чертежа на выбранную цель.

Каждая проверка даёт одну строку одного из трёх уровней:
    ok   — зелёный: хватает с запасом;
    warn — жёлтый: на грани, возможна нехватка;
    bad  — красный: точно не хватит или нужной системы нет вовсе.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from . import mission as ms
from .craft import G0, Vessel
from .i18n import L, u_ms
from .world import World

ROOT = Path(__file__).resolve().parent.parent
GENOME_FILE = ROOT / "data" / "best_genome.json"

# Какие вехи обучения доказывают, что пилот умеет эту цель
PROOF = {
    "LEO": ("stable_orbit",),
    "HEO": ("stable_orbit",),
    "LLO": ("mun_orbit",),
    "HLO": ("mun_orbit",),
    "LND": ("target_landing",),
}


@dataclass
class Check:
    status: str          # ok / warn / bad
    name: str
    text: str
    hint: str = ""


@dataclass
class Allocation:
    """Как бюджет Δv раскладывается по ступеням чертежа."""
    per_leg: list[tuple[ms.Maneuver, list[int], float]] = field(default_factory=list)
    shortfall: float = 0.0
    leftover: float = 0.0
    fuel_needed: float = 0.0          # LF+Ox, единиц
    fuel_available: float = 0.0
    landing_stage_twr: float | None = None
    stage_of_leg: dict[str, int] = field(default_factory=dict)


@dataclass
class Report:
    target: ms.Target
    budget: ms.Budget
    vessel: Vessel | None
    checks: list[Check] = field(default_factory=list)
    allocation: Allocation | None = None
    dv_available: float = 0.0

    @property
    def worst(self) -> str:
        levels = [c.status for c in self.checks]
        if "bad" in levels:
            return "bad"
        if "warn" in levels:
            return "warn"
        return "ok"


def load_genome() -> dict:
    """Профиль полёта из установленного модуля весов FlightProfile."""
    from .knowledge import genome
    return genome()


def fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


# ==========================================================================
def allocate(vessel: Vessel, budget: ms.Budget, world: World) -> Allocation:
    """Проходит по бюджету манёвр за манёвром и «сжигает» ступени."""
    stages = vessel.stages()
    alloc = Allocation(fuel_available=sum(s.propellant_units for s in stages))
    moon_g = ms.target_body(world, budget.target).g0
    i = 0
    m_cur = stages[0].m0 if stages else 0.0
    for leg in budget.legs:
        need = leg.dv
        used_stages = []
        while need > 1e-6 and i < len(stages):
            st = stages[i]
            m_cur = min(m_cur, st.m0) if m_cur else st.m0
            if m_cur <= st.m1 + 1e-9:
                i += 1
                if i < len(stages):
                    m_cur = stages[i].m0
                continue
            isp = st.isp_vac
            if leg.atmospheric and i == 0 and st.isp_asl:
                isp = st.isp_asl * ms.DV_ASCENT_ATM_SHARE + st.isp_vac * (1 - ms.DV_ASCENT_ATM_SHARE)
            can = isp * G0 * math.log(m_cur / st.m1)
            take = min(can, need)
            m_new = m_cur / math.exp(take / (isp * G0))
            burned_share = (m_cur - m_new) / max(st.m0 - st.m1, 1e-9)
            alloc.fuel_needed += burned_share * st.propellant_units
            used_stages.append(st.stage)
            alloc.stage_of_leg[leg.key] = st.stage
            if leg.key == "land":
                alloc.landing_stage_twr = st.thrust_vac / (m_cur * moon_g)
            m_cur = m_new
            need -= take
            if need > 1e-6:
                i += 1
                if i < len(stages):
                    m_cur = stages[i].m0
        alloc.per_leg.append((leg, used_stages, need))
        alloc.shortfall += max(0.0, need)
    # Остаток: что осталось в недожжённых ступенях (в вакууме)
    left = 0.0
    for j in range(i, len(stages)):
        st = stages[j]
        m0 = min(m_cur, st.m0) if j == i and m_cur else st.m0
        if m0 > st.m1:
            left += st.isp_vac * G0 * math.log(m0 / st.m1)
    alloc.leftover = left
    return alloc


# ==========================================================================
def analyse(world: World, target: ms.Target, vessel: Vessel | None) -> Report:
    budget = ms.budget(world, target, vessel)
    rep = Report(target, budget, vessel)
    if vessel is None:
        rep.checks.append(Check("bad", L("Blueprint", "Чертёж"),
                                L("not found: save the rocket in the VAB or put it on the pad",
                                  "не найден: сохраните ракету в VAB или выведите её на стартовый стол")))
        return rep

    h = world.home_body
    m = ms.target_body(world, target)
    ms_ = u_ms()
    stages = vessel.stages()
    alloc = allocate(vessel, budget, world)
    rep.allocation = alloc
    need = budget.total
    have = need - alloc.shortfall if alloc.shortfall > 0 else need + alloc.leftover
    rep.dv_available = have
    add = rep.checks.append

    # --- Δv ---------------------------------------------------------------
    if not stages:
        add(Check("bad", "Δv", L("no stage with an engine and fuel",
                                 "нет ни одной ступени с двигателем и топливом")))
    else:
        text = f"{fmt(have)} / {fmt(need)} {ms_}"
        reserve = (have / need - 1) * 100 if need else 0
        if have >= budget.total_margin:
            add(Check("ok", "Δv", f"{text}  " + L(f"(margin +{reserve:.0f} %)",
                                                   f"(запас +{reserve:.0f} %)")))
        elif have >= need:
            add(Check("warn", "Δv", f"{text}  " + L(
                f"(margin only +{reserve:.0f} %, {fmt(budget.total_margin)} recommended)",
                f"(запас всего +{reserve:.0f} %, рекомендуется {fmt(budget.total_margin)})")))
        else:
            add(Check("bad", "Δv", f"{text}  " + L(f"({fmt(need - have)} {ms_} short)",
                                                    f"(не хватает {fmt(need - have)} {ms_})")))

    # --- Топливо ----------------------------------------------------------
    if stages:
        text = f"{fmt(alloc.fuel_available)} / {fmt(alloc.fuel_needed)} " + L("u", "ед.")
        # Ступень сжигается целиком, поэтому «нужно» — это всё топливо
        # ступеней, которые уйдут до цели, плюс доля последней. Запас
        # оценивается по Δv: единицы топлива верхних ступеней «дороже».
        name = L("Fuel (LF+Ox)", "Топливо (LF+Ox)")
        if alloc.shortfall > 0:
            add(Check("bad", name, f"{text}  " + L("— runs out before the target",
                                                   "— кончится до цели")))
        elif have >= budget.total_margin:
            add(Check("ok", name, text))
        else:
            add(Check("warn", name, f"{text}  " + L("— reserve below 15 %",
                                                    "— резерв меньше 15 %")))

    # --- TWR --------------------------------------------------------------
    if stages:
        twr0 = stages[0].twr(h.g0, atm=h.atmosphere_depth > 0)
        genome = load_genome().get("genome", {}).get("design", {})
        wanted = max(1.2, float(genome.get("liftoff_twr", 1.3)) * 0.85)
        name = L("Liftoff TWR", "TWR на старте")
        text = f"{twr0:.2f}  " + L(f"(need ≥ {wanted:.2f})", f"(нужно ≥ {wanted:.2f})")
        if twr0 >= wanted:
            add(Check("ok", name, text))
        elif twr0 >= 1.0:
            add(Check("warn", name, f"{text} " + L("— slow climb, big losses",
                                                   "— медленный подъём, большие потери")))
        else:
            add(Check("bad", name, f"{twr0:.2f} " + L("— the rocket will not leave the pad",
                                                     "— ракета не оторвётся от стола")))

        for leg, used, _ in alloc.per_leg:
            if not used or leg.key in ("ascent", "land", "mcc"):
                continue
            st = next((s for s in stages if s.stage == used[0]), None)
            if st is None:
                continue
            twr = st.thrust_vac / (st.m0 * h.g0)
            if twr < leg.twr_min * 0.5:
                add(Check("warn", f"TWR · {leg.key.upper()}",
                          f"{twr:.2f} " + L("— long burn, expect losses",
                                            "— импульс растянется, возможны потери")))

    if target.landing:
        twr_l = alloc.landing_stage_twr
        name = L(f"Landing TWR ({m.name})", f"TWR посадки ({m.name})")
        if twr_l is None:
            add(Check("bad", name, L("no fuelled stage left for landing",
                                     "до посадки не останется ступени с топливом")))
        elif twr_l >= 2.0:
            add(Check("ok", name, f"{twr_l:.2f}  (g = {m.g0:.2f} " + L("m/s²", "м/с²") + ")"))
        elif twr_l >= 1.3:
            add(Check("warn", name, f"{twr_l:.2f} " + L("— soft landing is marginal",
                                                        "— мягкая посадка на пределе")))
        else:
            add(Check("bad", name, f"{twr_l:.2f} " + L("— not enough thrust to stop",
                                                       "— не хватит тяги погасить скорость")))

    # --- Управление -------------------------------------------------------
    cmd = vessel.count("command")
    name = L("Control", "Управление")
    if cmd == 0:
        add(Check("bad", name, L("no command module or probe core",
                                 "нет командного модуля или ядра зонда")))
    else:
        who = (L(f"crew {vessel.crew}", f"экипаж {vessel.crew}") if vessel.crewed
               else L("probe", "зонд"))
        add(Check("ok", name, L(f"command modules: {cmd} ({who})",
                                f"командных модулей: {cmd} ({who})")))

    wheels, rcs = vessel.count("reaction_wheel"), vessel.count("rcs")
    gimbal = any(p.gimbal > 0 for p in vessel.parts)
    name = L("Attitude", "Ориентация")
    if wheels:
        add(Check("ok", name, L(f"reaction wheels {wheels}", f"маховиков {wheels}")
                  + (f", RCS {rcs}" if rcs else "")
                  + (L(", gimbal", ", качание сопла") if gimbal else "")))
    elif rcs or gimbal:
        add(Check("warn", name, L("no reaction wheels: slow turns without thrust",
                                  "нет маховиков: вне тяги аппарат будет разворачиваться медленно")
                  + (L(" (RCS present)", " (есть RCS)") if rcs else "")))
    else:
        add(Check("bad", name, L("nothing to control attitude with",
                                 "нечем управлять ориентацией")))

    # --- Электропитание ---------------------------------------------------
    cap = vessel.ec_capacity
    rate = vessel.consumption + (0.01 if wheels else 0.0)   # маховики тратят только при повороте
    gen = vessel.solar_rate * 0.6 + vessel.generator_rate
    duration = budget.duration
    night = (m.period_at(ms.work_altitude(world, target) or ms.leo_altitude(h)) / 2
             if target.body != h.name else h.period_at(ms.leo_altitude(h)) / 2)
    name = L("Electricity", "Электричество")
    if cap <= 0:
        add(Check("bad", name, L("no batteries — control will be lost",
                                 "нет аккумуляторов — управление пропадёт")))
    elif gen >= rate:
        need_ec = rate * night * 1.5
        if cap >= need_ec:
            add(Check("ok", name, L(f"{fmt(cap)} EC, generation +{gen:.2f} EC/s",
                                    f"{fmt(cap)} EC, генерация +{gen:.2f} EC/с")))
        else:
            add(Check("warn", name, f"{fmt(cap)} / {fmt(need_ec)} EC "
                      + L("— possible shortage in shadow", "— возможная нехватка в тени")))
    else:
        need_ec = rate * duration * 1.2
        if cap >= need_ec:
            add(Check("ok", name, f"{fmt(cap)} / {fmt(need_ec)} EC "
                      + L("(no generation)", "(без генерации)")))
        elif cap >= need_ec * 0.7:
            add(Check("warn", name, f"{fmt(cap)} / {fmt(need_ec)} EC "
                      + L("— possible shortage", "— возможная нехватка")))
        else:
            add(Check("bad", name, f"{fmt(cap)} / {fmt(need_ec)} EC " + L(
                "— will run flat on the way, add panels or batteries",
                "— разрядится в пути, нужны панели или батареи")))

    # --- Связь (CommNet) --------------------------------------------------
    st = world.settings
    name = L("Comms", "Связь")
    if not st.commnet:
        add(Check("ok", name, L("CommNet is disabled in world settings",
                                "CommNet отключён в настройках мира")))
    else:
        distance = ms.comm_distance(world, target)
        power = vessel.best_antenna * st.range_modifier
        if power <= 0:
            add(Check("bad" if st.require_signal else "warn", name,
                      L("no antenna", "нет антенны")
                      + (L(" — control without signal is forbidden",
                           " — без сигнала управление запрещено") if st.require_signal else "")))
        else:
            rng = math.sqrt(power * st.dsn_power)
            text = L(f"range {fmt(rng / 1000)} km / need {fmt(distance / 1000)} km",
                     f"дальность {fmt(rng / 1000)} км / нужно {fmt(distance / 1000)} км")
            relay = vessel.count("relay")
            if rng >= distance * 1.2:
                far_side = (target.body != h.name and target.altitude < m.radius
                            and st.require_signal and not relay)
                if far_side:
                    add(Check("warn", name, text + L("; no signal over the Moon's far side",
                                                     "; над обратной стороной Луны сигнала не будет")))
                else:
                    add(Check("ok", name, text))
            elif rng >= distance * 0.8:
                add(Check("warn", name, text + L(" — link is marginal", " — связь на пределе")))
            else:
                add(Check("bad" if st.require_signal else "warn", name,
                          text + L(" — signal will be lost", " — сигнал пропадёт")))

    # --- Посадочные опоры -------------------------------------------------
    if target.landing:
        legs = vessel.count("leg")
        name = L("Landing legs", "Посадочные опоры")
        if legs >= 3:
            add(Check("ok", name, L(f"{legs} pcs", f"{legs} шт.")))
        elif legs > 0:
            add(Check("warn", name, L(f"{legs} pcs — may tip over",
                                      f"{legs} шт. — аппарат может опрокинуться")))
        else:
            add(Check("bad", name, L("none — the landing will be a crash",
                                     "нет — посадка закончится ударом о грунт")))
        if m.atmosphere_depth > 0:
            chutes = vessel.count("parachute")
            name = L("Parachutes", "Парашюты")
            if chutes >= 2:
                add(Check("ok", name, L(f"{chutes} pcs for the {m.name} atmosphere",
                                        f"{chutes} шт. для атмосферы {m.name}")))
            elif chutes:
                add(Check("warn", name, L(f"{chutes} pc — the touchdown may be too fast",
                                          f"{chutes} шт. — касание может быть слишком быстрым")))
            else:
                add(Check("warn", name, L(f"none — {m.name} has an atmosphere, landing on engines only",
                                          f"нет — у {m.name} атмосфера, садиться придётся на двигателях")))

    # --- Профиль автопилота (обученные веса) ------------------------------
    gen_data = load_genome()
    milestones = set(gen_data.get("milestones", []))
    proof = PROOF.get(target.code, ())
    name = L("Autopilot profile", "Профиль автопилота")
    if not gen_data:
        add(Check("bad", name, L("no flight weights installed — see Weights / Knowledge",
                                 "веса полёта не установлены — раздел «Веса / Знания»")))
    elif all(p in milestones for p in proof):
        add(Check("ok", name, L(f"trained, best score {gen_data.get('score', 0):.0f}",
                                f"обучен, лучший счёт {gen_data.get('score', 0):.0f}")))
    else:
        add(Check("warn", name, L("this target has not been passed in training yet",
                                  "эта цель в обучении ещё не пройдена")))

    if vessel.unknown_parts:
        add(Check("warn", L("Parts catalog", "Каталог деталей"),
                  L(f"{len(vessel.unknown_parts)} parts not found — estimate is approximate",
                    f"{len(vessel.unknown_parts)} дет. не найдено — расчёт приблизителен")))
    return rep
