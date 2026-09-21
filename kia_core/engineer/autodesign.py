"""Автономный конструктор: от постановки задачи до чертежа.

Никаких заранее заданных схем ракеты. Порядок работы такой же, какой
применил бы живой инженер:

1. РАСЧЁТ БЮДЖЕТА. По задаче (куда лететь, на какую орбиту, садиться ли,
   возвращаться ли) считается требуемая характеристическая скорость:
   круговая скорость v = sqrt(mu/r), потери на подъём, гомановский переход,
   торможение у цели по патч-коническому приближению.

2. РАЗБИЕНИЕ НА СТУПЕНИ. Бюджет режется на ступени, каждой назначается
   свой dV и свой минимальный TWR (стартовой — по земной гравитации и
   давлению, верхним — по вакууму).

3. ПОДБОР ДЕТАЛЕЙ. Для каждой ступени снизу вверх решается обратная задача
   Циолковского: перебираются все двигатели и баки каталога игры, для каждой
   пары ищется минимальное число баков, дающее нужный dV, проверяется TWR,
   выбирается вариант минимальной стартовой массы.

4. ОБОСНОВАНИЕ. Каждое решение сопровождается формулой и числами — они
   уходят в kia_design_report.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from ..config import G0
from ..logging_setup import get_logger
from ..mission_spec import MissionSpec
from . import rocket_math as rm
from .craft_writer import Blueprint, BoosterBuild, StageBuild
from .parts_catalog import PartCatalog, PartInfo, get_catalog
from .parts_db import get_body

log = get_logger("engineer.autodesign")

# КРУПНЫЕ БАКИ ЛУЧШЕ КУЧИ МЕЛКИХ.
#
# Оператор попросил прямо: «надо, чтобы он предпочитал большие топливные
# баки, а не много маленьких». Прямой штраф за число деталей для этого не
# годится — проверено: при `part_penalty` 1.0 вместо 0.35 ракета
# становится вдвое тяжелее (74.8 → 188.4 т) и деталей в ней БОЛЬШЕ.
#
# Работает ограничение сверху: конструктор просто вынужден взять бак
# покрупнее. Замер на задаче «сесть на Муну»:
#
#     лимит 12 -> 74.8 т, 53 детали, первая ступень 5 × FL-TX1800
#     лимит  4 -> 78.2 т, 52 детали, первая ступень 3 × Rockomax X200-32
#
# Плюс три тонны массы и на одну деталь меньше — размен честный.
MAX_TANKS_PER_STAGE = 4
# Вытянутость ступени (длина / поперечник), после которой начинается
# штраф. Четыре калибра — обычная пропорция ракетного блока; всё, что
# длиннее, гнётся и раскачивается.
SLENDER_FREE = 4.0
SLENDER_PENALTY = 0.08
# Больше восьми ускорителей вокруг одного блока в KSP не разместить без
# наложения обшивок, а связка становится неуправляемой по крену.
MAX_BOOSTERS = 8
# Меньше четырёх ускорителей не навешиваем: пара даёт симметрию только по
# одной оси, а четыре — по обеим.
MIN_BOOSTERS = 4
# Ускорители тяжелее полутора масс центрального блока — признак того, что
# ракету надо перепроектировать целиком, а не обвешивать.
MAX_BOOSTER_MASS_SHARE = 1.5
ATMOSPHERIC_LOSSES = 1150.0     # м/с, подъём сквозь атмосферу Кербина
AIRLESS_LOSSES = 250.0          # м/с, подъём с безатмосферного тела


# ==========================================================================
# Политика проектирования — то, что настраивает обучение
# ==========================================================================
@dataclass
class DesignPolicy:
    """Предпочтения конструктора. Именно их оптимизирует эволюция.

    Раньше эволюция перебирала индексы деталей; теперь она задаёт инженерную
    политику, а детали конструктор подбирает сам под задачу.
    """
    ascent_split: float = 0.55        # доля dV подъёма, отданная 1-й ступени
    liftoff_twr: float = 1.55         # целевой стартовый TWR
    upper_twr: float = 0.75           # минимальный TWR разгонных ступеней
    transfer_twr: float = 0.45        # минимальный TWR перелётной ступени
    dv_margin: float = 1.10           # запас dV сверх расчётного бюджета
    mass_penalty: float = 1.0         # вес массы при выборе (>1 — экономим массу)
    part_penalty: float = 0.35        # вес числа деталей (штраф за «этажерки»)
    science_level: float = 1.0        # 0..1 — сколько приборов брать
    solar_level: float = 1.0          # 0..1 — брать ли панели
    fin_level: float = 0.6            # 0..1 — аэродинамическая стабилизация
    # 0..1 — насколько охотно конструктор перекладывает старт на боковые
    # ускорители. На нуле ракета отрывается от стола одним центральным
    # блоком (та самая «сосиска»), выше — часть стартовой тяговооружённости
    # отдаётся навесным твердотопливникам, а центральный блок берётся
    # скромнее. Что выгоднее — решает эволюция по итогам полётов.
    booster_level: float = 0.5
    # Разрешены ли боковые ускорители вообще. НЕ входит в геном и
    # эволюцией не перебирается — это рубильник, а не предпочтение.
    #
    # Выключен по итогам живых замеров. Статистика за вечер:
    #
    #     с ускорителями:   4 запуска, средний счёт -658
    #     без ускорителей: 29 запусков, средний счёт   +5
    #
    # Четыре провала из четырёх, причём три из них — разрушение аппарата.
    # В журнале видно «Отделение ступени 0 -> 0» через тринадцать секунд
    # после отрыва: автостейджинг прогоняет ракету через все стадии до
    # нуля и подрывает её. Похоже, нумерация стадий связки конфликтует с
    # условием отстрела по остатку топлива (`should_stage` смотрит
    # SolidFuel в группе разделения), но точная причина не найдена.
    #
    # Расчёт связки, сборка и отчёт остаются рабочими и проверены
    # тестами — включается одной строкой, когда стадии будут разобраны.
    # ВКЛЮЧЕНЫ ОБРАТНО: причина прежних провалов найдена и устранена.
    #
    # Она была не в связке и не в чертеже — стадии там расставлены как в
    # штатных чертежах игры (замер по `УРНА-3`: ускоритель istg=2 на
    # радиальном разделителе istg=1, центральный двигатель istg=2).
    # Ломалось решение о моменте отстрела: `should_stage` смотрел остаток
    # топлива в группе разделения, а в ней вперемешку твердотопливные
    # шашки и жидкое топливо центрального блока. Шашки выгорают первыми,
    # условие срабатывает — и ракета делится, хотя центральный двигатель
    # ещё тянет. Теперь признак физический: пока хоть один работающий
    # двигатель даёт тягу, делить нечего.
    allow_boosters: bool = True
    # Скидка деталям установленных дополнений при выборе (0..0.35).
    # Не запрет и не приказ: сдвигает выбор в близких случаях, чтобы
    # купленное дополнение реально участвовало в конструкции.
    expansion_bonus: float = 0.12

    @staticmethod
    def bounds() -> dict[str, tuple[float, float]]:
        return {
            "ascent_split": (0.35, 0.75),
            "liftoff_twr": (1.25, 2.10),
            "upper_twr": (0.45, 1.30),
            "transfer_twr": (0.30, 1.00),
            # ПОТОЛОК ПОДНЯТ ПО ЗАМЕРУ, А НЕ ПО РАССУЖДЕНИЮ.
            #
            # Лучший живой полёт (#123) дошёл до апоапсиса 80.5 км и на
            # скруглении отработал 1526 м/с из потребных 1796 — не хватило
            # 270 м/с, орбита не замкнулась (Pe = −168 км). Запас при этом
            # уже стоял у прежнего потолка: эволюция держала 1.28…1.33 в
            # каждом из 25 последних запусков, то есть упиралась в него.
            # Расчётные потери 1150 м/с для такой мелкой и лобастой ракеты
            # занижены — недостачу и покрывает поднятый потолок.
            #
            # Противовес: нижняя граница 1.02 на месте, а `mass_penalty`
            # штрафует лишнюю массу — раздуться безнаказанно нельзя.
            "dv_margin": (1.02, 1.55),
            "mass_penalty": (0.5, 2.0),
            "part_penalty": (0.0, 1.5),
            "science_level": (0.0, 1.0),
            "solar_level": (0.0, 1.0),
            "fin_level": (0.0, 1.0),
            "booster_level": (0.0, 1.0),
            "expansion_bonus": (0.0, 0.35),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DesignPolicy":
        known = {}
        for key, value in data.items():
            if key not in cls.__dataclass_fields__:
                continue
            # allow_boosters — рубильник, а не число
            known[key] = bool(value) if key == "allow_boosters" else float(value)
        return cls(**known)

    def clamp(self) -> "DesignPolicy":
        for name, (lo, hi) in self.bounds().items():
            setattr(self, name, max(lo, min(hi, float(getattr(self, name)))))
        return self


# ==========================================================================
# Бюджет delta-V
# ==========================================================================
@dataclass
class BudgetLeg:
    name: str
    dv: float
    formula: str
    numbers: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeltaVBudget:
    legs: list[BudgetLeg] = field(default_factory=list)
    margin: float = 1.10

    @property
    def raw_total(self) -> float:
        return sum(leg.dv for leg in self.legs)

    @property
    def total(self) -> float:
        return self.raw_total * self.margin

    def to_dict(self) -> dict:
        return {"legs": [l.to_dict() for l in self.legs],
                "margin": self.margin,
                "raw_total": round(self.raw_total, 1),
                "total": round(self.total, 1)}


def compute_budget(spec: MissionSpec, margin: float = 1.10) -> DeltaVBudget:
    """Считает требуемый dV миссии из орбитальной механики, а не из таблицы."""
    home = get_body(spec.home_body)
    budget = DeltaVBudget(margin=margin)

    # --- выведение на опорную орбиту ---
    r_park = home.radius + spec.park_orbit_altitude
    v_circ = math.sqrt(home.mu / r_park)
    losses = ATMOSPHERIC_LOSSES if home.atmosphere_height > 0 else AIRLESS_LOSSES

    # ВЫСОКАЯ ОРБИТА ДОРОЖЕ НИЗКОЙ, А НЕ ДЕШЕВЛЕ.
    #
    # Считалось `sqrt(mu/r) + потери` прямо на целевой высоте. Круговая
    # скорость с высотой ПАДАЕТ, и бюджет выходил тем меньше, чем выше
    # орбита: на 800 км он потребовал 5195 м/с против 6451 на 100 км.
    # Ракета строилась под заниженную цифру, топливо кончалось на 48 км —
    # два полёта подряд упали, не выйдя из атмосферы.
    #
    # На деле путь такой: сперва низкая орбита (её скорость и потери —
    # главная статья), затем гомановский подъём до опорной высоты.
    r_low = home.radius + max(70_000.0, home.atmosphere_height + 10_000.0)
    v_low = math.sqrt(home.mu / r_low)
    budget.legs.append(BudgetLeg(
        f"Выведение на низкую орбиту {(r_low - home.radius)/1000:.0f} км ({home.name})",
        v_low + losses,
        "dV = sqrt(mu / r_низк) + потери(грав. + аэродин.)",
        f"sqrt({home.mu:.4g} / {r_low:.0f}) = {v_low:.0f} м/с; "
        f"потери ≈ {losses:.0f} м/с"))

    if r_park > r_low + 1000.0:
        a_raise = (r_low + r_park) / 2.0
        dv_raise = (math.sqrt(home.mu * (2.0 / r_low - 1.0 / a_raise)) - v_low
                    + v_circ - math.sqrt(home.mu * (2.0 / r_park - 1.0 / a_raise)))
        budget.legs.append(BudgetLeg(
            f"Подъём до опорной орбиты {spec.park_orbit_altitude/1000:.0f} км",
            dv_raise,
            "два импульса Гомана: разгон в перицентре + скругление в апоцентре",
            f"с {(r_low - home.radius)/1000:.0f} км до "
            f"{spec.park_orbit_altitude/1000:.0f} км = {dv_raise:.0f} м/с"))

    if spec.needs_transfer:
        target = get_body(spec.target_body)

        # МЕЖПЛАНЕТНЫЙ ПЕРЕЛЁТ СЧИТАЕТСЯ ВОКРУГ СВЕТИЛА.
        #
        # Прежняя формула брала μ Кербина и «радиус орбиты цели», то есть
        # обращалась с Дюной как со спутником Кербина. Для Муны это верно,
        # для планеты — нет: получалось 260 м/с там, где нужно около 950.
        # Та же подмена жила и в пилоте (см. transfer.py).
        interplanetary = (target.parent or spec.home_body) != spec.home_body
        if interplanetary:
            star = get_body(target.parent or "Kerbol")
            r1, r2 = home.orbit_radius, target.orbit_radius
            a_h = (r1 + r2) / 2.0
            v_home = math.sqrt(star.mu / r1)
            v_dep = math.sqrt(star.mu * (2.0 / r1 - 1.0 / a_h))
            v_inf_out = abs(v_dep - v_home)
            dv_inject = (math.sqrt(v_inf_out ** 2 + 2.0 * home.mu / r_park)
                         - v_circ)
            budget.legs.append(BudgetLeg(
                f"Сход к {target.name} с опорной орбиты",
                dv_inject,
                "v_p = sqrt(v_inf^2 + 2*mu_дома/r);  dV = v_p - v_круг",
                f"v_inf = {v_inf_out:.0f} м/с (гоман вокруг {star.name}); "
                f"итог {dv_inject:.0f} м/с"))
            if spec.needs_capture:
                v_arr = math.sqrt(star.mu * (2.0 / r2 - 1.0 / a_h))
                v_target_orb = math.sqrt(star.mu / r2)
                v_inf_in = abs(v_target_orb - v_arr)
                rp = target.radius + spec.target_orbit_altitude
                dv_capture = (math.sqrt(v_inf_in ** 2 + 2.0 * target.mu / rp)
                              - math.sqrt(target.mu / rp))
                budget.legs.append(BudgetLeg(
                    f"Торможение у {target.name} на "
                    f"{spec.target_orbit_altitude/1000:.0f} км",
                    dv_capture,
                    "v_p = sqrt(v_inf^2 + 2*mu/r_p);  dV = v_p - sqrt(mu/r_p)",
                    f"v_inf = {v_inf_in:.0f} м/с; итог {dv_capture:.0f} м/с"))
            if spec.needs_landing:
                v_surface = math.sqrt(target.mu
                                      / (target.radius + spec.target_orbit_altitude))
                budget.legs.append(BudgetLeg(
                    f"Посадка на {target.name}", v_surface * 1.15,
                    "гашение орбитальной скорости с запасом 15%",
                    f"{v_surface:.0f} × 1.15 = {v_surface*1.15:.0f} м/с"))
            budget.legs.append(BudgetLeg(
                "Запас на коррекции курса", 120.0,
                "межпланетный перелёт длится сотни суток",
                "две-три правки по 40-60 м/с"))
            return budget

        r_target = target.orbit_radius
        a_t = (r_park + r_target) / 2.0
        v_peri = math.sqrt(home.mu * (2.0 / r_park - 1.0 / a_t))
        dv_inject = v_peri - v_circ
        budget.legs.append(BudgetLeg(
            f"Перелётный импульс к {target.name}",
            dv_inject,
            "dV = sqrt(mu*(2/r1 - 1/a)) - sqrt(mu/r1),  a = (r1+r2)/2",
            f"a = ({r_park:.0f}+{r_target:.0f})/2 = {a_t:.0f} м; "
            f"{v_peri:.0f} - {v_circ:.0f} = {dv_inject:.0f} м/с"))

        if spec.needs_capture:
            v_apo = math.sqrt(home.mu * (2.0 / r_target - 1.0 / a_t))
            v_target_orb = math.sqrt(home.mu / r_target)
            v_inf = abs(v_target_orb - v_apo)
            rp = target.radius + spec.target_orbit_altitude
            v_hyper = math.sqrt(v_inf ** 2 + 2 * target.mu / rp)
            v_circ_target = math.sqrt(target.mu / rp)
            dv_capture = v_hyper - v_circ_target
            budget.legs.append(BudgetLeg(
                f"Торможение у {target.name} на {spec.target_orbit_altitude/1000:.0f} км",
                dv_capture,
                "v_p = sqrt(v_inf^2 + 2*mu/r_p);  dV = v_p - sqrt(mu/r_p)",
                f"v_inf = {v_inf:.0f} м/с; v_p = {v_hyper:.0f}; "
                f"v_круг = {v_circ_target:.0f}; итог {dv_capture:.0f} м/с"))

        if spec.needs_landing:
            v_surface = math.sqrt(target.mu / (target.radius + spec.target_orbit_altitude))
            dv_land = v_surface * 1.15
            budget.legs.append(BudgetLeg(
                f"Посадка на {target.name}",
                dv_land,
                "dV ≈ 1.15 * v_круг (гашение скорости + гравитационные потери)",
                f"1.15 * {v_surface:.0f} = {dv_land:.0f} м/с"))
            if spec.needs_return:
                budget.legs.append(BudgetLeg(
                    f"Взлёт с {target.name}", v_surface * 1.10,
                    "dV ≈ 1.10 * v_круг", f"{v_surface * 1.10:.0f} м/с"))

        if spec.needs_return:
            budget.legs.append(BudgetLeg(
                f"Возвращение к {home.name}",
                abs(dv_inject) * 0.45 + 200.0,
                "dV ≈ импульс выхода из SOI цели + коррекция",
                "аэродинамическое торможение в атмосфере считаем бесплатным"))

    budget.legs.append(BudgetLeg(
        "Коррекции и маневрирование", 120.0,
        "инженерный резерв на неточности наведения",
        "120 м/с"))
    return budget


# ==========================================================================
# Разбиение на ступени
# ==========================================================================
@dataclass
class StageRequirement:
    index: int                 # 0 — нижняя
    role: str
    dv_required: float
    min_twr: float
    gravity: float
    pressure: float            # среднее давление на участке, атм
    twr_body: str

    def to_dict(self) -> dict:
        return asdict(self)


def plan_stages(spec: MissionSpec, budget: DeltaVBudget,
                policy: DesignPolicy) -> list[StageRequirement]:
    """Режет бюджет на ступени. Число ступеней следует из задачи, не из шаблона."""
    home = get_body(spec.home_body)
    ascent_dv = budget.legs[0].dv * budget.margin
    rest = budget.total - ascent_dv

    reqs: list[StageRequirement] = []
    split = max(0.35, min(0.75, policy.ascent_split))
    reqs.append(StageRequirement(
        index=0, role="Стартовая ступень (атмосферный участок)",
        dv_required=ascent_dv * split, min_twr=policy.liftoff_twr,
        gravity=home.surface_gravity, pressure=0.45, twr_body=home.name))
    reqs.append(StageRequirement(
        index=1, role="Разгонная ступень (выход на орбиту)",
        dv_required=ascent_dv * (1.0 - split), min_twr=policy.upper_twr,
        gravity=home.surface_gravity, pressure=0.03, twr_body=home.name))

    if not spec.needs_transfer:
        # Задача «только орбита»: остаток бюджета (коррекции) отдаём верхней
        # ступени — третья ступень ради 100 м/с не нужна.
        reqs[1].dv_required += max(0.0, rest)
    elif rest > 50.0:
        if spec.needs_landing:
            target = get_body(spec.target_body)
            land_dv = sum(l.dv for l in budget.legs
                          if "Посадка" in l.name or "Взлёт" in l.name) * budget.margin
            transfer_dv = max(0.0, rest - land_dv)
            reqs.append(StageRequirement(
                index=2, role="Перелётная ступень",
                dv_required=transfer_dv, min_twr=policy.transfer_twr,
                gravity=home.surface_gravity, pressure=0.0, twr_body=home.name))
            reqs.append(StageRequirement(
                index=3, role=f"Посадочная ступень ({target.name})",
                dv_required=land_dv, min_twr=1.6,
                gravity=target.surface_gravity, pressure=0.0, twr_body=target.name))
        else:
            reqs.append(StageRequirement(
                index=2, role="Перелётная ступень",
                dv_required=rest, min_twr=policy.transfer_twr,
                gravity=home.surface_gravity, pressure=0.0, twr_body=home.name))

    # индексация: 0 — нижняя, поэтому список переворачиваем «сверху вниз» при решении
    return reqs


# ==========================================================================
# Решение одной ступени
# ==========================================================================
@dataclass
class StageSolution:
    requirement: StageRequirement
    engine: PartInfo
    tank: PartInfo
    tank_count: int
    decoupler: PartInfo | None
    payload_mass: float
    candidates_examined: int = 0
    rationale: list[str] = field(default_factory=list)

    # ---- массы ----
    @property
    def tanks_dry(self) -> float:
        return self.tank.dry_mass * self.tank_count

    @property
    def fuel_mass(self) -> float:
        return self._tank_fuel * self.tank_count

    _tank_fuel: float = 0.0

    @property
    def structure_mass(self) -> float:
        return (self.engine.dry_mass + self.tanks_dry
                + (self.decoupler.dry_mass if self.decoupler else 0.0))

    @property
    def wet_mass(self) -> float:
        return self.payload_mass + self.structure_mass + self.fuel_mass

    @property
    def dry_mass(self) -> float:
        return self.payload_mass + self.structure_mass

    # ---- характеристики ----
    def isp(self) -> float:
        return self.engine.isp_at(self.requirement.pressure)

    def thrust(self) -> float:
        return self.engine.thrust_at(self.requirement.pressure)

    def delta_v(self) -> float:
        return rm.delta_v(self.isp(), self.wet_mass, self.dry_mass)

    def twr(self) -> float:
        return (self.thrust() * 1000.0) / (self.wet_mass * 1000.0 * self.requirement.gravity)

    def burn_time(self) -> float:
        return rm.burn_time(self.isp(), self.thrust(), self.wet_mass, self.dry_mass)

    def summary(self) -> dict:
        return {
            "index": self.requirement.index,
            "role": self.requirement.role,
            "engine": self.engine.title,
            "engine_part": self.engine.name,
            "thrust_kn": round(self.thrust(), 1),
            "isp_s": round(self.isp(), 1),
            "tank": self.tank.title,
            "tank_part": self.tank.name,
            "tank_count": self.tank_count,
            "payload_mass_t": round(self.payload_mass, 3),
            "dry_mass_t": round(self.dry_mass, 3),
            "wet_mass_t": round(self.wet_mass, 3),
            "fuel_mass_t": round(self.fuel_mass, 3),
            "dv_required": round(self.requirement.dv_required, 1),
            "dv_actual": round(self.delta_v(), 1),
            "twr_required": round(self.requirement.min_twr, 2),
            "twr_actual": round(self.twr(), 2),
            "burn_time_s": round(self.burn_time(), 1),
            "candidates_examined": self.candidates_examined,
            "rationale": self.rationale,
        }


class StageSolver:
    """Обратная задача Циолковского: подбор двигателя и баков под dV и TWR."""

    def __init__(self, catalog: PartCatalog, policy: DesignPolicy):
        self.catalog = catalog
        self.policy = policy

    def solve(self, req: StageRequirement, payload_mass: float,
              need_decoupler: bool,
              require_gimbal: bool = False) -> StageSolution | None:
        """Подбор ступени.

        `require_gimbal` — для ступени, которая работает в атмосфере.
        Двигатель без качающегося сопла оставляет аппарат без управления
        по тангажу: держать курс будет нечем, кроме маховика ядра.

        Это не теория. Замер живой игры: конструктор выбрал T-1 Dart
        (качание 0.0°), автопилот просил тангаж 90°, аппарат стоял на 70°
        при нулевом угле атаки — то есть команду он просто не отрабатывал.
        Рядом в каталоге лежал LV-T45 Swivel с качанием 3° и почти той же
        тягой, но угол сопла в подборе не участвовал вовсе.
        """
        best: StageSolution | None = None
        best_cost = float("inf")
        examined = 0
        # Ослабляем требование, если ни один двигатель нужного размера
        # качанием не обладает — лучше собрать хоть что-то.
        if require_gimbal and not any(
                e.gimbal > 0 for d in self.catalog.diameters()
                for e in self.catalog.engines(diameter=d)):
            require_gimbal = False

        for diameter in self.catalog.diameters():
            engines = self.catalog.engines(diameter=diameter)
            tanks = self.catalog.tanks(diameter=diameter)
            decoupler = None
            if need_decoupler:
                decouplers = self.catalog.decouplers(diameter=diameter)
                if not decouplers:
                    continue
                decoupler = decouplers[0]
            if not engines or not tanks:
                continue

            for engine in engines:
                isp = engine.isp_at(req.pressure)
                thrust = engine.thrust_at(req.pressure)
                if isp <= 0 or thrust <= 0:
                    continue
                if require_gimbal and engine.gimbal <= 0:
                    continue
                for tank in tanks:
                    tank_fuel = tank.resource_mass(self.catalog.densities)
                    if tank_fuel <= 0:
                        continue
                    fixed = (payload_mass + engine.dry_mass
                             + (decoupler.dry_mass if decoupler else 0.0))
                    count = self._tanks_needed(req.dv_required, isp, fixed,
                                               tank.dry_mass, tank_fuel)
                    examined += 1
                    if count is None:
                        continue
                    wet = fixed + count * (tank.dry_mass + tank_fuel)
                    twr = (thrust * 1000.0) / (wet * 1000.0 * req.gravity)
                    if twr < req.min_twr:
                        continue
                    # избыточная тяга на старте = лишние потери на сопротивление
                    excess = max(0.0, twr - max(req.min_twr, 1.8)) if req.index == 0 else 0.0
                    cost = (wet * self.policy.mass_penalty
                            + count * self.policy.part_penalty
                            + excess * 0.5)

                    # ШИРЕ, А НЕ ДЛИННЕЕ.
                    #
                    # Подбор шёл только по массе, и выигрывал всегда самый
                    # мелкий бак: поставить его четыре раза дешевле по
                    # сухой массе, чем один крупный. Ракета выходила
                    # «сосиской» — оператор описал это точно: она гибкая,
                    # её всё время кренит, и автопилот весь подъём
                    # занимается не выведением, а выпрямлением.
                    #
                    # Штраф берётся за ВЫТЯНУТОСТЬ ступени — отношение её
                    # длины к поперечнику. До четырёх калибров ступень
                    # штрафа не платит вовсе: это нормальная пропорция.
                    # Дальше цена растёт, и один широкий бак начинает
                    # выигрывать у четырёх узких, как и должно быть.
                    caliber = max(0.1, tank.hull_diameter)
                    slender = (count * tank.height) / caliber
                    if slender > SLENDER_FREE:
                        cost += wet * SLENDER_PENALTY * (slender - SLENDER_FREE)
                    # Скидка деталям дополнения. Оператор поставил Making
                    # History и хочет видеть его в ракетах, а по чистой
                    # массе стоковые часто выигрывают на копейки: разница
                    # в десятки килограммов решает выбор, хотя двигатели
                    # дополнения заметно лучше по импульсу (Wolfhound —
                    # 380 с против 345 у лучшего стокового).
                    #
                    # Скидка НЕ отменяет расчёт: она сдвигает выбор в
                    # близких случаях, а деталь, проигрывающую заметно,
                    # не спасёт. Ноль означает «выбирать строго по массе».
                    # Качающееся сопло — это управляемость, и она стоит
                    # немного массы. Скидка мягкая: она решает спор между
                    # близкими двигателями, а заведомо худший не спасёт.
                    if engine.gimbal > 0:
                        cost -= wet * min(0.10, engine.gimbal * 0.015)

                    bonus = self.policy.expansion_bonus
                    if bonus > 0.0:
                        newer = sum(1 for p in (engine, tank)
                                    if "SquadExpansion" in (p.cfg_path or ""))
                        cost -= wet * bonus * (newer / 2.0)
                    if cost < best_cost:
                        solution = StageSolution(
                            requirement=req, engine=engine, tank=tank,
                            tank_count=count, decoupler=decoupler,
                            payload_mass=payload_mass)
                        solution._tank_fuel = tank_fuel
                        best, best_cost = solution, cost

        if best is None:
            log.warning("Ступень %d (%s): решение не найдено (dV=%.0f, TWR≥%.2f)",
                        req.index, req.role, req.dv_required, req.min_twr)
            return None

        best.candidates_examined = examined
        best.rationale = self._explain(best, examined)
        log.info("Ступень %d: %s + %d×%s -> dV=%.0f (нужно %.0f), TWR=%.2f (нужно %.2f)",
                 req.index, best.engine.title, best.tank_count, best.tank.title,
                 best.delta_v(), req.dv_required, best.twr(), req.min_twr)
        return best

    # ------------------------------------------------------------------
    @staticmethod
    def _tanks_needed(dv_required: float, isp: float, fixed_mass: float,
                      tank_dry: float, tank_fuel: float) -> int | None:
        """Минимальное число баков, дающее нужный dV (перебор по n)."""
        ve = isp * G0
        for n in range(1, MAX_TANKS_PER_STAGE + 1):
            dry = fixed_mass + n * tank_dry
            wet = dry + n * tank_fuel
            if dry <= 0:
                continue
            if ve * math.log(wet / dry) >= dv_required:
                return n
        return None

    def _explain(self, s: StageSolution, examined: int) -> list[str]:
        req = s.requirement
        ve = s.isp() * G0
        return [
            f"Перебрано {examined} сочетаний «двигатель × бак» из каталога игры.",
            f"Требование ступени: dV ≥ {req.dv_required:.0f} м/с при TWR ≥ "
            f"{req.min_twr:.2f} (g={req.gravity:.2f} м/с², p={req.pressure:.2f} атм).",
            f"Выбран {s.engine.title}: тяга {s.thrust():.0f} кН, Isp {s.isp():.0f} с "
            f"(v_e = Isp·g₀ = {ve:.0f} м/с), сухая масса {s.engine.dry_mass:.2f} т.",
            f"Баки: {s.tank_count} × {s.tank.title} — топливо {s.fuel_mass:.2f} т, "
            f"конструкция {s.tanks_dry:.2f} т.",
            f"Циолковский: dV = {ve:.0f}·ln({s.wet_mass:.2f}/{s.dry_mass:.2f}) = "
            f"{s.delta_v():.0f} м/с — запас {s.delta_v() - req.dv_required:+.0f} м/с.",
            f"TWR = F/(m·g) = {s.thrust():.0f}·1000/({s.wet_mass:.2f}·1000·"
            f"{req.gravity:.2f}) = {s.twr():.2f}.",
            f"Время работы на полной тяге: {s.burn_time():.0f} с.",
        ]


# ==========================================================================
# Полный проект
# ==========================================================================
@dataclass
class PayloadChoice:
    part: PartInfo
    role: str
    reason: str
    radial: bool = False
    fin: bool = False
    clamp: bool = False           # пусковая мачта: держит ракету на столе
    nose: bool = False            # носовой обтекатель на макушку
    bay: bool = False             # прибор прячется ВНУТРЬ служебного отсека
    service_bay: bool = False     # сам служебный отсек
    fairing: bool = False         # основание защитного обтекателя
    avionics: bool = False        # маховик в стеке под ядром
    stack: bool = False           # деталь корпуса аппарата: стоит В колонне


@dataclass
class BoosterPack:
    """Просчитанная связка боковых ускорителей центрального блока."""
    part: PartInfo
    count: int
    decoupler: PartInfo | None
    nose: PartInfo | None
    thrust_each: float          # кН на уровне моря
    wet_each: float             # т, заправленный
    dry_each: float             # т
    burn_time: float            # с
    delta_v_bonus: float        # м/с, добавка к бюджету — в план НЕ засчитана
    core_twr: float             # TWR одного центрального блока
    twr: float                  # TWR связки целиком
    rationale: list[str] = field(default_factory=list)

    @property
    def total_wet(self) -> float:
        return self.wet_each * self.count

    @property
    def total_thrust(self) -> float:
        return self.thrust_each * self.count

    def to_dict(self) -> dict:
        return {
            "part": self.part.name, "title": self.part.title,
            "count": self.count,
            "thrust_each_kn": round(self.thrust_each, 1),
            "total_thrust_kn": round(self.total_thrust, 1),
            "wet_each_t": round(self.wet_each, 3),
            "total_wet_t": round(self.total_wet, 3),
            "burn_time_s": round(self.burn_time, 1),
            "delta_v_bonus": round(self.delta_v_bonus, 1),
            "twr_core_only": round(self.core_twr, 3),
            "twr_with_boosters": round(self.twr, 3),
            "decoupler": self.decoupler.name if self.decoupler else None,
            "nose": self.nose.name if self.nose else None,
            "rationale": self.rationale,
        }


@dataclass
class VehicleDesign:
    spec: MissionSpec
    policy: DesignPolicy
    budget: DeltaVBudget
    payload: list[PayloadChoice]
    stages: list[StageSolution]
    blueprint: Blueprint
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    boosters: BoosterPack | None = None

    @property
    def viable(self) -> bool:
        return not self.problems

    @property
    def payload_mass(self) -> float:
        return sum(c.part.dry_mass for c in self.payload)

    @property
    def total_mass(self) -> float:
        core = self.stages[0].wet_mass if self.stages else self.payload_mass
        return core + (self.boosters.total_wet if self.boosters else 0.0)

    @property
    def total_delta_v(self) -> float:
        return sum(s.delta_v() for s in self.stages)

    @property
    def launch_twr(self) -> float:
        """Стартовая тяговооружённость ВСЕГО, что стоит на столе.

        Считать только по центральному блоку нельзя: с ускорителями это
        разные числа, и именно суммарное решает, оторвётся ракета или нет.
        """
        if self.boosters is not None:
            return self.boosters.twr
        return self.stages[0].twr() if self.stages else 0.0

    @property
    def part_count(self) -> int:
        return self.blueprint.part_count

    def summary(self) -> dict:
        return {
            "mission": self.spec.describe(),
            "budget": self.budget.to_dict(),
            "policy": self.policy.to_dict(),
            "total_mass_t": round(self.total_mass, 3),
            "payload_mass_t": round(self.payload_mass, 3),
            "total_delta_v": round(self.total_delta_v, 1),
            "required_delta_v": round(self.budget.total, 1),
            "launch_twr": round(self.launch_twr, 3),
            "part_count": self.part_count,
            "viable": self.viable,
            "problems": self.problems,
            "warnings": self.warnings,
            "payload": [{"part": c.part.name, "title": c.part.title,
                         "role": c.role, "reason": c.reason} for c in self.payload],
            "stages": [s.summary() for s in self.stages],
            "boosters": self.boosters.to_dict() if self.boosters else None,
        }


class AutonomousEngineer:
    """Конструктор, который сам считает и сам выбирает детали."""

    # Потолки массы для «взять с запасом». Не ограничение сверху ради
    # экономии, а граница здравого смысла: за ними идут детали для
    # межпланетных станций, которым на орбите Кербина делать нечего.
    LANDING_LEGS = 4
    RCS_BLOCKS = 4
    RCS_ROLE = "рулевой блок"
    WHEEL_MASS_CAP = 0.25
    LEG_ROLE = "посадочная опора"
    LEG_MASS_CAP = 0.15           # т — сюда попадает LT-2, самая крупная стойка
    BATTERY_MASS_CAP = 0.06       # т — сюда попадает Z-400 (400 ЕЭ)
    ANTENNA_MASS_CAP = 0.20       # т — сюда попадает RA-5 (2 Гм, не раскрывается)

    def __init__(self, catalog: PartCatalog | None = None):
        self.catalog = catalog or get_catalog()

    # ------------------------------------------------------------------
    def choose_payload(self, spec: MissionSpec, policy: DesignPolicy) -> list[PayloadChoice]:
        """Полезная нагрузка БЕСПИЛОТНОГО аппарата.

        KIA строит только беспилотники: наверху всегда зондовое ядро
        категории Command. Живой экипаж не берётся никогда — управление и
        SAS обеспечивает ядро, а не кербонавт. Отсюда жёсткие требования:
        ядро обязано иметь запас электричества, а к нему обязательно
        добавляются батарея и солнечные панели, иначе аппарат обесточится
        и станет неуправляемым («Нет управления»).
        """
        choices: list[PayloadChoice] = []

        cores = self.catalog.probe_cores()
        if not cores:
            raise RuntimeError("В каталоге игры не найдено ни одного зондового ядра")
        # Ядру нужен управляющий момент: без маховика автопилот не удержит
        # курс на выведении, и полёт закончится потерей устойчивости.
        steerable = [c for c in cores if "ModuleReactionWheel" in c.modules
                     and c.resources.get("ElectricCharge", 0) >= 5]
        core = (steerable or cores)[0]
        has_wheel = "ModuleReactionWheel" in core.modules
        choices.append(PayloadChoice(
            core, "командный модуль",
            f"беспилотное ядро управления: {core.dry_mass:.3f} т, "
            f"{core.resources.get('ElectricCharge', 0):.0f} ЕЭ на борту, "
            f"диаметр {core.diameter:g} м; "
            + ("есть маховик — автопилоту есть чем держать курс"
               if has_wheel else
               "БЕЗ маховика: управление только газом и качанием сопла")))

        # Питание обязательно: без энергии зонд теряет управление
        batteries = [b for b in self.catalog.batteries() if b.can_surface_attach]
        if batteries:
            # ЁМКОСТЬ БЕРЁМ С ЗАПАСОМ, А НЕ ПО МИНИМУМУ МАССЫ.
            #
            # Прежде брался первый по списку — Z-100 на 100 ЕЭ. Замер
            # живого полёта к Муне: заряд упал со 110 ЕЭ до нуля на
            # перелёте, зонд обесточился и не смог повернуться к узлу
            # торможения. Ста единиц не хватает на манёвры в тени и на
            # работу маховика при долгом наведении.
            battery = max(
                (b for b in batteries if b.dry_mass <= self.BATTERY_MASS_CAP),
                key=lambda b: b.resources.get("ElectricCharge", 0.0),
                default=batteries[0])
            choices.append(PayloadChoice(
                battery, "аккумулятор",
                f"буфер энергии {battery.resources.get('ElectricCharge', 0):.0f} ЕЭ "
                f"на манёвры в тени планеты и долгие развороты",
                radial=True, bay=True))
        panels = [p for p in self.catalog.solar_panels()
                  if p.can_surface_attach and p.dry_mass < 0.1]
        if panels:
            # БЕРЁМ ПОВОРОТНУЮ, А НЕ САМУЮ ЛЁГКУЮ.
            #
            # Список отсортирован по массе, и первой шла неподвижная
            # накладка OX-STAT. Она заряжает, только если случайно
            # смотрит на солнце, а ракета в полёте смотрит на узел.
            # Замер живого полёта к Муне: обе панели раскрыты, поток
            # 0.00, заряд упал со 110 ЕЭ до нуля — зонд обесточился,
            # перестал слушаться руля и не смог затормозиться у цели.
            # В игре это выглядело как «Нет связи с зондом».
            tracking = [p for p in panels if p.sun_tracking]
            panel = (max(tracking, key=lambda p: p.charge_rate)
                     if tracking else panels[0])
            count = 2 if policy.solar_level > 0.4 else 1
            note = ("поворачивается за солнцем" if panel.sun_tracking
                    else "неподвижная — заряд зависит от разворота аппарата")
            for _ in range(count):
                choices.append(PayloadChoice(
                    panel, "солнечная панель",
                    f"постоянная подзарядка ({panel.charge_rate:.2f} ЕЭ/с, "
                    f"{note}): беспилотник без энергии неуправляем",
                    radial=True))

        # ВНУТРЬ ОТСЕКА — ТОЛЬКО НЕРАСКРЫВАЮЩИЕСЯ.
        #
        # Самая дальняя антенна в игре, «Коммунотрон 88-88» на 100 Гм, —
        # раскрывающаяся тарелка размахом больше самой ракеты. Спрятанная
        # в служебный отсек, она разворачивалась прямо сквозь корпус:
        # оператор увидел в ангаре жёлтый зонт вокруг носа. Внутри отсека
        # деталь не должна менять габарит.
        antennas = [a for a in self.catalog.antennas()
                    if a.can_surface_attach and a.antenna_power > 0
                    and not any("Deployable" in m for m in a.modules)]
        # СПУТНИКУ СВЯЗИ — РЕТРАНСЛЯТОР, А НЕ ПРОСТО АНТЕННА.
        #
        # Разбор эталона оператора «Спутник связи SWM-94»: там стоит
        # RelayAntenna100. Разница не в дальности, а в назначении: обычная
        # антенна говорит ТОЛЬКО с Кербином, ретранслятор пересылает чужой
        # сигнал. Созвездие из обычных антенн связи не даёт вовсе — каждый
        # аппарат будет молчать ровно там, где он и нужен: за горизонтом.
        if spec.is_relay:
            relays = [a for a in antennas if a.is_relay]
            if relays:
                # У ретранслятора запас берём по дальности без оглядки на
                # общий потолок массы антенны: RA-100 весит 0.65 т, но
                # ради него спутник и запускается.
                relay = max(relays, key=lambda a: a.antenna_power)
                choices.append(PayloadChoice(
                    relay, "ретранслятор",
                    f"ретранслятор {relay.title}: дальность "
                    f"{relay.antenna_power/1e9:.0f} Гм и ПЕРЕСЫЛКА чужого "
                    f"сигнала — то, ради чего спутник и висит на орбите",
                    radial=True))
                antennas = []      # вторую антенну ставить незачем

        if antennas:
            # ДАЛЬНОСТЬ ТОЖЕ С ЗАПАСОМ.
            #
            # Самая лёгкая антенна — «Коммунотрон 16» на 500 км. До Муны
            # 12 000 км, и связь держится лишь за счёт мощности наземной
            # станции. Любая цель дальше Муны такой антенне не по силам,
            # а без связи беспилотник неуправляем. Берём самую дальнюю в
            # пределах разумной массы: HG-55 даёт 15 Гм при 0.075 т.
            antenna = max(
                (a for a in antennas if a.dry_mass <= self.ANTENNA_MASS_CAP),
                key=lambda a: a.antenna_power, default=antennas[0])
            choices.append(PayloadChoice(
                antenna, "антенна",
                f"связь с ЦУП на {antenna.antenna_power/1e9:.0f} Гм: без неё "
                f"зонд теряет управление вне зоны видимости",
                radial=True, bay=True))

        # СПУТНИКУ СВЯЗИ ПАРАШЮТ НЕ НУЖЕН: он не возвращается никогда, а
        # лишняя масса наверху — это ΔV, отнятое у развода по точкам.
        # Зато нужен запас питания: половину витка он идёт в тени планеты,
        # а ретранслятор ест энергию непрерывно, в отличие от приборов.
        if spec.is_relay:
            choices = [c for c in choices
                       if c.role not in ("аккумулятор", "солнечная панель")]
            choices.extend(self.choose_satellite_body(spec))

        # ЛУНОХОДУ ПАРАШЮТ И ПОСАДОЧНЫЕ ОПОРЫ ЛИШНИЕ: он садится на
        # собственных колёсах, а спусковую ступень сбрасывает.
        if spec.is_rover:
            choices = [c for c in choices
                       if c.role not in ("аккумулятор", "солнечная панель",
                                         "парашют", self.LEG_ROLE)]
            choices.extend(self.choose_rover_body(spec))

        # Парашют вешаем радиально, чтобы ядро оставалось самой верхней деталью
        if not spec.is_relay and (spec.needs_return or spec.home_body == "Kerbin"):
            radial_chutes = sorted(
                (c for c in self.catalog.parachutes()
                 if c.can_surface_attach and c.bottom_node is None),
                key=lambda c: ("drogue" in c.name.lower(), c.dry_mass))
            stack_chutes = [c for c in self.catalog.parachutes() if c.bottom_node]
            if radial_chutes:
                chute = radial_chutes[0]
                choices.append(PayloadChoice(
                    chute, "парашют",
                    f"радиальное крепление {chute.title} — командный модуль "
                    f"остаётся самой верхней деталью аппарата", radial=True))
            elif stack_chutes:
                choices.append(PayloadChoice(
                    stack_chutes[0], "парашют",
                    f"торможение в атмосфере {spec.home_body}"))

        if spec.collect_science and policy.science_level > 0.2:
            count = 1 + int(policy.science_level * 2)
            science = [p for p in self.catalog.science()
                       if p.can_surface_attach and p.dry_mass <= 0.15]
            for part in science[:count]:
                choices.append(PayloadChoice(
                    part, "научный прибор",
                    f"сбор данных в полёте ({part.dry_mass:.3f} т)",
                    radial=True, bay=True))

        # Служебный отсек: приборы прячутся ВНУТРЬ.
        #
        # Барометр, термометр, аккумулятор и антенна, навешанные снаружи,
        # в стоковой аэродинамике KSP каждый набирает своё сопротивление
        # и свой нагрев. Внутри закрытого отсека они не обдуваются вовсе.
        # Солнечные панели внутрь НЕ прячем: закрытая оболочка перекрывает
        # им солнце, и они перестают давать ток — а обесточенный зонд уже
        # однажды стоил сорванного торможения у Муны.
        bay_parts = [c for c in choices if c.bay]
        if bay_parts:
            bays = [b for b in self.catalog.all()
                    if b.is_cargo_bay
                    and "ModuleProceduralFairing" not in b.modules
                    and b.top_node is not None and b.bottom_node is not None]
            # НУЖЕН ЦИЛИНДР, А НЕ КОНУС.
            #
            # Самым лёгким отсеком оказался SM-6A — переходник-конус,
            # сужающийся с 1.25 до 0.625 м. Приборы, расставленные по
            # окружности, вылезали сквозь его наклонную стенку: на снимке
            # из ангара видно аккумулятор и датчик, торчащие наружу.
            # У настоящего грузового отсека (`ModuleCargoBay`) полость
            # цилиндрическая, и внутри помещается то, что в неё ставят.
            # Цилиндр от конуса отличают РАЗМЕРЫ УЗЛОВ, а не профили: у
            # SM-6A профиль один (size1), но узлы разные — сверху 0,
            # снизу 1, то есть деталь сужается с 1.25 до 0.625 м. У
            # Service Bay оба узла размера 1.
            cylindrical = [b for b in bays
                           if b.top_node.size == b.bottom_node.size]
            pool = cylindrical or bays
            fitting = [b for b in pool
                       if abs(b.diameter - core.diameter) < 1e-6] or pool
            if fitting:
                bay = min(fitting, key=lambda b: b.dry_mass)
                choices.append(PayloadChoice(
                    bay, "служебный отсек",
                    f"{len(bay_parts)} прибор(ов) убраны внутрь: снаружи каждый "
                    f"набирает своё лобовое сопротивление и нагрев",
                    service_bay=True))
            else:
                # Отсека нет — приборы останутся снаружи, и это надо
                # сказать вслух, а не молча ухудшить аэродинамику.
                for c in bay_parts:
                    c.bay = False

        # УПРАВЛЕНИЕ ПО КРЕНУ: МАХОВИК И RCS.
        #
        # Качание сопла даёт тангаж и рысканье, но НЕ крен. Замер живого
        # полёта на 120 км:
        #
        #     момент тангаж/рысканье  31.77 кН·м
        #     момент по крену          0.50 кН·м
        #     вращение                 0.72 рад/с (41 °/с), не гаснет
        #     ошибка автопилота        92°
        #     заряд                    410 из 410 — питание НИ ПРИ ЧЁМ
        #
        # Полкилоньютон-метра против момента инерции 5200 кг·м² гасят
        # такую раскрутку семь секунд — если ничего ей не мешает. А ей
        # мешает работающий двигатель. Маховик даёт крен даром, RCS —
        # добавку и управление на посадке, где сопло уже не помощник.
        wheels = [w for w in self.catalog.all()
                  if w.roll_torque > 0 and not w.is_command
                  and w.top_node is not None and w.bottom_node is not None
                  and w.dry_mass <= self.WHEEL_MASS_CAP]
        if wheels:
            wheel = max(wheels, key=lambda w: w.roll_torque)
            choices.append(PayloadChoice(
                wheel, "маховик",
                f"управляющий момент {wheel.roll_torque:.0f} кН·м по крену: "
                f"качание сопла крен не даёт вовсе",
                avionics=True))

        thrusters = [t for t in self.catalog.all()
                     if "ModuleRCSFX" in t.modules or "ModuleRCS" in t.modules]
        thrusters = [t for t in thrusters if t.can_surface_attach
                     and t.dry_mass <= 0.05 and not t.is_engine]
        tanks = [t for t in self.catalog.all()
                 if t.resources.get("MonoPropellant", 0) >= 20
                 and t.can_surface_attach and t.dry_mass <= 0.05]
        if thrusters and tanks:
            block = max(thrusters, key=lambda t: t.dry_mass)
            tank = max(tanks, key=lambda t: t.resources["MonoPropellant"])
            for _ in range(self.RCS_BLOCKS):
                choices.append(PayloadChoice(
                    block, self.RCS_ROLE,
                    "рулевой блок RCS: крен и точное маневрирование там, "
                    "где качание сопла бессильно", radial=True))
            choices.append(PayloadChoice(
                tank, "бак монотоплива",
                f"{tank.resources['MonoPropellant']:.0f} единиц для RCS",
                radial=True, bay=True))

        # ПОСАДОЧНЫЕ ОПОРЫ — ровно четыре, как в образцах.
        #
        # Замер по чертежам из поставки игры и по чертежам оператора:
        # `Two-Stage Lander`, `Super-Heavy Lander`, `PT Series Munsplorer`
        # и `«Наполлон»` — везде ровно четыре опоры, радиально у низа
        # посадочной ступени. Три дают неустойчивый треножник на склоне,
        # шесть — лишняя масса.
        if spec.needs_landing:
            legs = sorted((p for p in self.catalog.all()
                           if "ModuleWheelDeployment" in p.modules
                           or "ModuleLandingLeg" in p.modules),
                          key=lambda p: p.dry_mass)
            legs = [p for p in legs if p.can_surface_attach
                    and "Gear" not in p.name and "roverWheel" not in p.name]
            if legs:
                # САМАЯ КРУПНАЯ СТОЙКА, А НЕ ВТОРАЯ ПО ЛЁГКОСТИ.
                #
                # Замер по снимку оператора: на LT-1 (0.05 т) аппарат стоял
                # на срезе сопла — стойка короче двигателя и до грунта не
                # доставала. В каталоге установки настоящих посадочных
                # стоек три: LT-05 (0.015), LT-1 (0.05), LT-2 (0.100).
                # Длина растёт вместе с массой, поэтому берём тяжелейшую.
                #
                # Противовес, чтобы «бери крупнейшую» не превратилось в
                # «тащи что угодно»: четыре опоры не вправе съесть больше
                # сотой доли стартовой массы.
                fitting = [p for p in legs if p.dry_mass <= self.LEG_MASS_CAP]
                leg = (fitting or legs)[-1]
                for _ in range(self.LANDING_LEGS):
                    choices.append(PayloadChoice(
                        leg, self.LEG_ROLE,
                        f"посадочная опора {leg.title}: четыре штуки, как в "
                        f"штатных посадочных модулях игры"))

        # ЗАЩИТНЫЙ ОБТЕКАТЕЛЬ вокруг всей полезной нагрузки.
        #
        # Всё, что торчит на верхнем блоке — панели, парашют, приборы, —
        # в стоковой аэродинамике KSP считается по отдельности и каждое
        # набирает своё сопротивление и нагрев. Створки закрывают их
        # целиком и сбрасываются за атмосферой, так что панелям под ними
        # солнце не нужно.
        if get_body(spec.home_body).atmosphere_height > 0:
            shells = [f for f in self.catalog.all()
                      if "ModuleProceduralFairing" in f.modules
                      and f.top_node is not None and f.bottom_node is not None]
            fitting = [f for f in shells
                       if abs(f.diameter - core.diameter) < 1e-6] or shells
            if fitting:
                shell = min(fitting, key=lambda f: f.dry_mass)
                choices.append(PayloadChoice(
                    shell, "защитный обтекатель",
                    f"створки укрывают полезную нагрузку целиком "
                    f"({shell.dry_mass:.3f} т) и сбрасываются за атмосферой",
                    fairing=True))

        # Носовой обтекатель. В стоковой аэродинамике KSP сопротивление
        # считается по каждой детали отдельно, и открытый верхний торец
        # получает полный коэффициент лобового сопротивления. Конус его
        # закрывает — на подъёме сквозь атмосферу это десятки-сотни м/с,
        # причём даром: сам конус весит граммы. Ставится только при
        # старте с тела с атмосферой: в вакууме он мёртвый груз.
        if get_body(spec.home_body).atmosphere_height > 0:
            cones = (self.catalog.nose_cones(diameter=core.diameter)
                     or self.catalog.nose_cones())
            if cones:
                cone = cones[0]
                choices.append(PayloadChoice(
                    cone, "носовой обтекатель",
                    f"закрывает верхний торец: {cone.dry_mass:.3f} т против "
                    f"полного лобового сопротивления тупой макушки",
                    nose=True))

        choices.extend(self.choose_control(spec, policy, core))
        choices.extend(self.choose_fins(spec, policy))
        choices.extend(self.choose_upper_fins(spec, policy))
        choices.extend(self.choose_clamps(spec))

        # ЛИШНЕЕ СНИМАЕТСЯ В КОНЦЕ, А НЕ ПО ХОДУ.
        #
        # Фильтр стоял в середине списка, и всё, что добавлялось ПОСЛЕ
        # него, проходило мимо: у лунохода остались и парашют, и четыре
        # посадочные стойки. Он садится на собственных колёсах, а
        # спусковую ступень сбрасывает — ни то, ни другое ему не нужно.
        if spec.is_rover:
            # НОСОВОЙ КОНУС И ГИРОДИН ЛУНОХОДУ НИ К ЧЕМУ, и оператор
            # спросил об этом прямо. Конус закрывает макушку РАКЕТЫ от
            # набегающего потока — но луноход и так едет под обтекателем,
            # а на грунте острый нос сверху просто смешон. Маховик держит
            # курс в полёте; машине на колёсах держать нечего, а весит он
            # больше всей научной части.
            choices = [c for c in choices
                       if c.role not in ("парашют", self.LEG_ROLE,
                                         "носовой обтекатель", "маховик",
                                         self.UPPER_FIN_ROLE)]
        return choices

    # ------------------------------------------------------------------
    # Состав спутника связи, снятый со снимка эталона оператора.
    SAT_BATTERIES = 3             # Z-1k в колонне корпуса
    SAT_PANELS = 2                # «Гигантор XL» по бортам, как крылья
    SAT_ENGINES = 4               # RV-1 «Юнец» вокруг корпуса

    def choose_satellite_body(self, spec: MissionSpec) -> list[PayloadChoice]:
        """Корпус спутника связи — по снимку оператора, а не по ракете.

        СПУТНИК СТРОИТСЯ ИНАЧЕ, ЧЕМ РАКЕТА, И ЭТО НЕ КОСМЕТИКА.
        Первая попытка собрала его обычной нагрузкой: батарейка Z-400
        сбоку, две накладки OX-10C, парашют. Оператор ответил снимком
        своего аппарата: тарелка на макушке, две панели-крыла в размах
        корпуса, батареи ВНУТРИ колонны, венец малых двигателей вокруг.
        Разница по существу: Z-1k — деталь стековая, сбоку её не
        повесить вовсе, а «Гигантор XL» даёт 24.4 ЕЭ/с против 0.35 у
        накладки — в семьдесят раз больше, и ретранслятору этого хватает
        с запасом.

        Двигатели RV-1 нужны спутнику для развода по своей точке
        созвездия: разгонная ступень выводит его на общую орбиту, а
        разойтись на 90° по фазе аппарат обязан сам.
        """
        out: list[PayloadChoice] = []

        banks = [b for b in self.catalog.batteries()
                 if b.top_node is not None and b.bottom_node is not None]
        if banks:
            # Z-1k НАЗВАН ОПЕРАТОРОМ ПОИМЕННО, и спорить тут не с чем:
            # по ёмкости на тонну Z-1k и Z-4K одинаковы (20 000 ЕЭ/т),
            # так что «крупнее» не значит «лучше» — значит только грубее
            # шаг. Три Z-1k дают 3000 ЕЭ и делятся по корпусу ровнее.
            named = [b for b in banks if "z-1k" in b.title.lower()]
            bank = (named[0] if named else
                    max(banks, key=lambda b: b.resources.get("ElectricCharge", 0.0)))
            for _ in range(self.SAT_BATTERIES):
                out.append(PayloadChoice(
                    bank, "батарея корпуса",
                    f"{bank.title}: {bank.resources.get('ElectricCharge', 0):.0f} ЕЭ "
                    f"в колонне корпуса — половину витка спутник идёт в тени, "
                    f"а ретранслятор ест непрерывно", stack=True))

        big = [p for p in self.catalog.solar_panels() if p.can_surface_attach]
        if big:
            panel = max(big, key=lambda p: p.charge_rate)
            for _ in range(self.SAT_PANELS):
                out.append(PayloadChoice(
                    panel, "солнечная панель",
                    f"{panel.title}: {panel.charge_rate:.1f} ЕЭ/с — крылья по "
                    f"бортам, как на эталоне", radial=True))

        verniers = [e for e in self.catalog.all()
                    if e.is_engine and e.can_surface_attach
                    and e.dry_mass <= 0.25]
        if verniers:
            engine = max(verniers, key=lambda e: e.dry_mass)
            # ДВИГАТЕЛЮ НУЖЕН БАК. Первая сборка дала спутнику четыре
            # RV-1 и ни капли топлива: развод по точкам созвездия был бы
            # невыполним, а масса — потрачена впустую.
            # ОБА КОМПОНЕНТА, А НЕ ОДИН. По одному лишь `LiquidFuel`
            # выбор упал на Mk1 Liquid Fuel Fuselage — самолётный бак без
            # окислителя. RV-1 на нём не запустится вовсе.
            tanks = [t for t in self.catalog.all()
                     if t.resources.get("LiquidFuel", 0) > 0
                     and t.resources.get("Oxidizer", 0) > 0
                     and t.top_node is not None and t.bottom_node is not None
                     and t.dry_mass <= 0.35 and not t.is_engine]
            if tanks:
                tank = max(tanks, key=lambda t: t.resources["LiquidFuel"])
                out.append(PayloadChoice(
                    tank, "бак спутника",
                    f"{tank.title}: топливо на развод по точке созвездия "
                    f"({tank.resources['LiquidFuel']:.0f} ед.)", stack=True))
            for _ in range(self.SAT_ENGINES):
                out.append(PayloadChoice(
                    engine, "двигатель спутника",
                    f"{engine.title}: венец малых двигателей — ими спутник "
                    f"сам разводится по своей точке созвездия", radial=True))
        return out

    # Состав лунохода, снятый с эталона «Вездеход со спусковым модулем».
    ROVER_WHEELS = 6              # три пары: на склоне устойчивее четырёх
    ROVER_LAMPS = 2               # ночь на Муне длится половину оборота
    ROVER_PANELS = 4
    ROVER_BATTERIES = 4
    ROVER_ROLE_WHEEL = "колесо"

    def choose_rover_body(self, spec: MissionSpec) -> list[PayloadChoice]:
        """Луноход по эталону оператора, а не по общей нагрузке.

        Разбор чертежа «Вездеход со спусковым модулем» (53 детали):
        палуба — структурная панель, под ней ШЕСТЬ колёс, сверху ядро
        `probeCoreOcto`, четыре батареи, три датчика, две антенны и
        восемь малых панелей. Всё, что выше разделителя — бак, тор и
        четыре радиальных двигателя, — спусковая ступень, и на грунте
        она сбрасывается.

        Шесть колёс, а не четыре: на склоне четырёхколёсный вездеход
        опрокидывается, и в эталоне их именно шесть.
        """
        out: list[PayloadChoice] = []

        deck = next((p for p in self.catalog.all()
                     if p.name == "structuralPanel1"), None)
        if deck is not None:
            out.append(PayloadChoice(
                deck, "палуба",
                f"{deck.title}: рама, на которой держится всё остальное",
                stack=True))

        wheels = [p for p in self.catalog.all()
                  if ("ModuleWheelMotor" in p.modules or "roverWheel" in p.name)
                  and p.can_surface_attach and p.dry_mass <= 0.15]
        if wheels:
            wheel = max(wheels, key=lambda p: p.dry_mass)
            for _ in range(self.ROVER_WHEELS):
                out.append(PayloadChoice(
                    wheel, self.ROVER_ROLE_WHEEL,
                    f"{wheel.title}: шесть колёс — на склоне четыре "
                    f"опрокидываются", radial=True))

        # ФОНАРЬ — ЭТО СВЕТ, А НЕ ШАССИ. У посадочных стоек и колёс тоже
        # есть `ModuleLight` (у них горят подсветки), и выбор «самый
        # тяжёлый из светящихся» дал LY-99 Extra Large Landing Gear —
        # шасси в четверть тонны вместо лампы в пять грамм.
        lamps = [p for p in self.catalog.all()
                 if "ModuleLight" in p.modules and p.can_surface_attach
                 and "ModuleWheelBase" not in p.modules
                 and "ModuleLandingLeg" not in p.modules
                 and p.dry_mass <= 0.02]
        if lamps:
            lamp = max(lamps, key=lambda p: p.dry_mass)
            for _ in range(self.ROVER_LAMPS):
                out.append(PayloadChoice(
                    lamp, "фонарь",
                    f"{lamp.title}: лунная ночь длится половину оборота, "
                    f"и без света ехать некуда", radial=True))

        drills = [p for p in self.catalog.all()
                  if "ModuleResourceHarvester" in p.modules
                  and p.can_surface_attach]
        if drills:
            drill = min(drills, key=lambda p: p.dry_mass)
            out.append(PayloadChoice(
                drill, "бур",
                f"{drill.title}: добыча грунта на месте", radial=True))

        batteries = [b for b in self.catalog.batteries() if b.can_surface_attach]
        if batteries:
            battery = max(batteries, key=lambda b: b.resources.get("ElectricCharge", 0.0)
                          if b.dry_mass <= self.BATTERY_MASS_CAP else -1)
            for _ in range(self.ROVER_BATTERIES):
                out.append(PayloadChoice(
                    battery, "аккумулятор",
                    f"{battery.title}: запас на ночь и на работу бура",
                    radial=True))

        panels = [p for p in self.catalog.solar_panels()
                  if p.can_surface_attach and p.dry_mass < 0.1]
        if panels:
            panel = max(panels, key=lambda p: p.charge_rate)
            for _ in range(self.ROVER_PANELS):
                out.append(PayloadChoice(
                    panel, "солнечная панель",
                    f"{panel.title}: подзарядка на ходу", radial=True))

        return out

    def choose_control(self, spec: MissionSpec, policy: DesignPolicy,
                       core: PartInfo) -> list[PayloadChoice]:
        """Органы управления по нормам, снятым с образцов.

        Раньше их не выбирали вовсе: аппарат довольствовался маховиком
        внутри зондового ядра, и этого хватало ровно до первого живого
        полёта. Замер: KIA просила тангаж 90°, ракета стояла на 70° при
        нулевом угле атаки — команда не отрабатывалась физически.

        Образцы из папки чертежей говорят иначе (см. reference_designs):
        даже четырёхтонный спутник «Зоркий» несёт три маховика, а
        шеститонный SWM-94 — два маховика и три сопла с качанием 22°.
        """
        from . import reference_designs as ref

        norms = ref.load()
        if not norms.known:
            return []

        # Оценка массы аппарата на этом этапе грубая — точной ещё нет,
        # ступени не подобраны. Для выбора класса «лёгкий/тяжёлый» хватает.
        rough_mass = ref.HEAVY_MASS if spec.needs_transfer else 10.0
        choices: list[PayloadChoice] = []

        need_wheels = max(0, norms.wheels_for(rough_mass) - 1)   # один в ядре
        wheels = self.catalog.reaction_wheels(diameter=core.diameter)             or self.catalog.reaction_wheels()
        if need_wheels and wheels:
            wheel = wheels[0]
            for _ in range(need_wheels):
                choices.append(PayloadChoice(
                    wheel, "маховик",
                    f"управляющий момент без расхода топлива: образцы такой "
                    f"массы несут {norms.wheels_for(rough_mass)} шт, в ядре "
                    f"есть только один"))

        if norms.wants_rcs(rough_mass):
            blocks = self.catalog.rcs_blocks()
            if blocks:
                block = blocks[min(2, len(blocks) - 1)]
                for _ in range(4):
                    choices.append(PayloadChoice(
                        block, "блок RCS",
                        f"образцы тяжелее {norms.rcs_from_mass:.0f} т без RCS "
                        f"не летают: маховика на такую массу не хватает",
                        radial=True))
        return choices

    # ------------------------------------------------------------------
    def choose_fins(self, spec: MissionSpec, policy: DesignPolicy) -> list[PayloadChoice]:
        """Аэродинамическая стабилизация нижней ступени.

        Ракета с тяжёлой головой и двигателем внизу статически неустойчива в
        атмосфере: центр давления оказывается выше центра масс, и аппарат
        разворачивает набегающим потоком. Стабилизаторы у среза нижнего бака
        смещают центр давления вниз. Нужны только при старте с тела с
        атмосферой."""
        home = get_body(spec.home_body)
        if home.atmosphere_height <= 0 or policy.fin_level <= 0.15:
            return []
        # Только ракетные стабилизаторы: самолётные крылья на борту ракеты
        # торчат вбок и сносят пусковые мачты (см. rocket_fins).
        fins = (self.catalog.rocket_fins(steerable=False)
                or self.catalog.rocket_fins())
        if not fins:
            return []

        # Уровень оперения выбирает не только ЧИСЛО стабилизаторов, но и их
        # РАЗМЕР. Восстанавливающий момент даёт ПЛОЩАДЬ, а не количество:
        # четыре пластинки Basic Fin по десять килограммов при fin_level на
        # потолке аппарат не удерживали — срывы 50-67° от потока.
        #
        # Площадь в каталоге не указана, но для стоковых аэродинамических
        # деталей масса ей прямо пропорциональна, а список отсортирован по
        # массе — значит индекс и есть выбор размера.
        #
        # Цена ошибки здесь высокая, и она уже дважды заплачена: самолётное
        # крыло Swept Wing Type A разнесло аппарат на 68 метрах, а AV-T1
        # упирался размахом в захват пусковой мачты (152 метра). Оба случая
        # закрыты — выбор сужен до настоящих ракетных стабилизаторов
        # (`rocket_fins`), а мачта разведена с оперением по высоте.
        #
        # Замер, ради которого всё это делалось: с AV-T1 выведение впервые
        # прошло целиком, 89° -> 32° без единого срыва, апоапсис 80.0 км.
        index = int(round(policy.fin_level * (len(fins) - 1)))
        fin = fins[max(0, min(len(fins) - 1, index))]
        count = 3 if policy.fin_level < 0.66 else 4
        reason = (f"устойчивость в атмосфере {home.name}: {count} шт по "
                  f"{fin.dry_mass:.3f} т смещают центр давления ниже центра "
                  f"масс (размер выбран по уровню оперения "
                  f"{policy.fin_level:.2f})")
        return [PayloadChoice(fin, "стабилизатор", reason, fin=True)
                for _ in range(count)]

    UPPER_FIN_ROLE = "стабилизатор верхней ступени"

    def choose_upper_fins(self, spec: MissionSpec,
                          policy: DesignPolicy) -> list[PayloadChoice]:
        """Аэродинамическая стабилизация ВЕРХНЕЙ ступени.

        Нижнее оперение уходит вместе с первой ступенью, а разделение по
        замеру живого полёта (#127) происходит на 20 км при напоре 14 кПа.
        За полторы секунды после сброса аппарат уходил от −6.5° по потоку
        до 42.6° и переворачивался через крен −172°. Так кончился 21 полёт
        из 25.

        Берётся САМЫЙ ЛЁГКИЙ ракетный стабилизатор и всего три штуки:
        верхней ступени нужно лишь пережить остаток атмосферы, и платить
        за это разгонной массой нельзя. Противовес правилу: при слабом
        оперении внизу (`fin_level` мал) наверху его не будет вовсе —
        это тот же признак «аэродинамика здесь не нужна».
        """
        home = get_body(spec.home_body)
        if home.atmosphere_height <= 0 or policy.fin_level <= 0.15:
            return []
        fins = (self.catalog.rocket_fins(steerable=False)
                or self.catalog.rocket_fins())
        if not fins:
            return []
        fin = fins[0]
        reason = (f"верхняя ступень остаётся без оперения нижней и на "
                  f"разделении внутри атмосферы {home.name} переворачивается "
                  f"потоком: 3 шт по {fin.dry_mass:.3f} т держат её по потоку")
        return [PayloadChoice(fin, self.UPPER_FIN_ROLE, reason, fin=True)
                for _ in range(3)]

    # ------------------------------------------------------------------
    def choose_boosters(self, spec: MissionSpec, policy: DesignPolicy,
                        core: StageSolution) -> BoosterPack | None:
        """Навесные ускорители: добрать стартовую тягу, не раздувая блок.

        Зачем они нужны. Стартовая тяговооружённость — самое дорогое
        требование ко всей ракете: чтобы поднять её одним центральным
        двигателем, приходится брать двигатель заведомо избыточный, и
        его мёртвую массу потом тащат на орбиту все ступени. Боковой
        твердотопливник даёт ту же тягу за долю сухой массы и, главное,
        уходит через полторы минуты — дальше ракета летит налегке.

        Как считается число. Нужно наименьшее ЧЁТНОЕ N, при котором

            (F_центр + N·F_уск) / ((m_центр + N·m_уск)·g) ≥ TWR_цель

        Решение существует не всегда: если сам ускоритель тяжелее, чем
        может поднять собственная тяга при целевом TWR (F_уск ≤
        TWR·g·m_уск), никакое N не поможет — каждый следующий только
        топит связку. Такие кандидаты отбрасываются сразу.
        """
        if not policy.allow_boosters or policy.booster_level <= 0.05:
            return None
        req = core.requirement
        g = req.gravity
        target = policy.liftoff_twr
        core_thrust = core.thrust()
        core_mass = core.wet_mass
        core_twr = core_thrust * 1000.0 / (core_mass * 1000.0 * g)
        # СВЯЗКА — ЗАМЫСЕЛ, А НЕ ЗАПЛАТКА.
        #
        # Прежде ускорители появлялись, только если центральный блок сам
        # не отрывался от стола. Но конструктор всегда строит его
        # самодостаточным, поэтому связки не было ни разу.
        #
        # Оператор предложил обратное и прав: пусть тягу на старте даёт
        # связка, а первая ступень станет меньше. Мёртвая масса связки
        # уходит через полторы минуты, а укороченный центральный блок
        # летит дальше и не гнётся, как «сосиска».
        if core_twr >= target and policy.booster_level < 0.5:
            return None                 # тяги хватает, ускорители — балласт

        candidates = self.catalog.solid_boosters()
        if not candidates:
            log.info("Ускорителей в каталоге нет — старт на одном центральном блоке")
            return None
        densities = self.catalog.densities

        best: BoosterPack | None = None
        best_thrust = 0.0
        for part in candidates:
            thrust = part.thrust_at(req.pressure)
            wet = part.wet_mass(densities)
            if thrust <= 0 or wet <= 0:
                continue
            gain = thrust - target * g * wet     # вклад одного в баланс
            if gain <= 0:
                continue                        # сам себя не поднимает
            need = target * g * core_mass - core_thrust
            n = math.ceil(need / gain)
            # ЧЕТЫРЕ, А НЕ ПАРА, И ПОКРУПНЕЕ.
            #
            # Пары мелких «Блох» хватало по числам, но связка из двух
            # коротышек и держит хуже, и выглядит несерьёзно: оператор
            # сразу сказал, что ускорителей должно быть четыре и больше
            # размером. Четыре штуки вокруг блока — это ещё и симметрия
            # по обеим осям, а не только по одной.
            n = max(MIN_BOOSTERS, n + (n % 2))
            if n > MAX_BOOSTERS:
                continue
            added = n * wet
            if added > core_mass * MAX_BOOSTER_MASS_SHARE:
                continue                        # хвост тяжелее собаки
            # Из подходящих берём САМЫЙ МОЩНЫЙ, а не самый лёгкий: прежнее
            # правило «наименьшая добавленная масса» всегда приводило к
            # мелочи, потому что мелочь и весит меньше.
            if thrust > best_thrust + 1e-6:
                best_thrust = thrust
                isp = part.isp_at(req.pressure)
                burn = (part.resource_mass(densities) * 1000.0
                        / (thrust * 1000.0 / (isp * G0))) if isp > 0 else 0.0
                twr = ((core_thrust + n * thrust) * 1000.0
                       / ((core_mass + added) * 1000.0 * g))
                # Прирост dV считается по связке в целом и в бюджет НЕ
                # засчитывается: ускорители работают одновременно с
                # центральным блоком, точный расчёт параллельной схемы
                # сложнее, а завышать бюджет опаснее, чем занижать.
                wet_all = core_mass + added
                dry_all = wet_all - n * part.resource_mass(densities)
                bonus = isp * G0 * math.log(wet_all / dry_all) if dry_all > 0 else 0.0
                best = BoosterPack(
                    part=part, count=n,
                    decoupler=(self.catalog.radial_decouplers() or [None])[0],
                    nose=self._booster_nose(part),
                    thrust_each=thrust, wet_each=wet, dry_each=part.dry_mass,
                    burn_time=burn, delta_v_bonus=bonus,
                    core_twr=core_twr, twr=twr)

        if best is None:
            log.info("Подходящей связки ускорителей не нашлось: центральный "
                     "блок TWR=%.2f при цели %.2f", core_twr, target)
            return None

        best.rationale = [
            f"Центральный блок сам даёт TWR {core_twr:.2f} при цели "
            f"{target:.2f} — тяги на отрыв не хватает.",
            f"Выбран {best.part.title}: тяга {best.thrust_each:.0f} кН на "
            f"уровне моря при заправленной массе {best.wet_each:.2f} т.",
            f"Баланс одного ускорителя: {best.thrust_each:.0f} - {target:.2f}"
            f"·{g:.2f}·{best.wet_each:.2f} = "
            f"{best.thrust_each - target * g * best.wet_each:+.0f} кН в плюс.",
            f"Навешено {best.count} шт (только чётное число, строго "
            f"симметрично): суммарно {best.total_thrust:.0f} кН и "
            f"{best.total_wet:.2f} т.",
            f"TWR связки = ({core_thrust:.0f}+{best.total_thrust:.0f})·1000/"
            f"(({core_mass:.2f}+{best.total_wet:.2f})·1000·{g:.2f}) = "
            f"{best.twr:.2f}.",
            f"Горят {best.burn_time:.0f} с, после чего сбрасываются "
            f"радиальными разделителями. Прибавка dV ≈ "
            f"{best.delta_v_bonus:.0f} м/с в бюджет не засчитана — это запас.",
        ]
        log.info("Ускорители: %d × %s -> TWR %.2f (было %.2f), +%.1f т",
                 best.count, best.part.title, best.twr, core_twr, best.total_wet)
        return best

    def _booster_nose(self, booster: PartInfo) -> PartInfo | None:
        """Конус на макушку ускорителя: тупой торец в потоке дорого стоит."""
        if booster.top_node is None:
            return None
        exact = self.catalog.nose_cones(diameter=booster.diameter)
        return (exact or self.catalog.nose_cones() or [None])[0]

    def choose_clamps(self, spec: MissionSpec) -> list[PayloadChoice]:
        """Пусковые мачты: удержать аппарат на столе до зажигания.

        Высокая ракета малого диаметра стоит на площадке фактически на
        одном сочленении, и в KSP этого хватает, чтобы она качнулась и
        завалилась ЕЩЁ ДО СТАРТА. Наблюдение живой игры, 06.08.

        Пилот тут бессилен, и учить его бесполезно: в тренажёре аппарат
        на столе удерживается вертикально до самого отрыва
        (`ground_contact`), то есть такого отказа сеть не видела ни разу.
        Чему не показывают, тому не научишься — значит, решать должен
        конструктор.

        Четыре мачты по 0.1 т, отстреливаются на зажигании нижней
        ступени. В полёте не участвуют, на расчёт ΔV и TWR не влияют.
        """
        clamps = self.catalog.launch_clamps()
        if not clamps:
            log.warning("Пусковых мачт в каталоге нет — ракета поедет со "
                        "стола как есть")
            return []
        clamp = clamps[0]
        count = 4
        reason = (f"удержание на столе до зажигания: {count} шт по "
                  f"{clamp.dry_mass:.3f} т, отстрел вместе с нижней ступенью")
        return [PayloadChoice(clamp, "пусковая мачта", reason, clamp=True)
                for _ in range(count)]

    # ------------------------------------------------------------------
    def design(self, spec: MissionSpec, policy: DesignPolicy | None = None) -> VehicleDesign:
        policy = (policy or DesignPolicy()).clamp()
        budget = compute_budget(spec, margin=policy.dv_margin)
        requirements = plan_stages(spec, budget, policy)
        payload = self.choose_payload(spec, policy)
        # Пусковые мачты отстреливаются на зажигании и никуда не летят —
        # считать их полезной нагрузкой нельзя. Замер: когда они попали в
        # массу, инженер начал строить ракету, способную поднять то, что
        # остаётся на площадке, и стартовая масса выросла с 7.4 до 14.5 т.
        payload_mass = sum(c.part.dry_mass for c in payload if not c.clamp)

        # Если конструктору разрешены ускорители, требование к стартовой
        # тяге центрального блока ослабляется: тягу доберут навесные
        # твердотопливники. Ниже 1.05 не опускаемся никогда — это
        # страховка на случай, если подходящей связки не найдётся:
        # ракета обязана уметь оторваться от стола сама по себе.
        if policy.booster_level > 0.05:
            for req in requirements:
                if req.index == 0:
                    # ПОПЫТКА УСИЛИТЬ СКИДКУ НИЧЕГО НЕ ДАЛА — ПРОВЕРЕНО.
                    #
                    # Хотелось, чтобы связка появлялась всегда: оператор
                    # просил четыре крупных ускорителя. Но размер
                    # центрального блока диктует не тяга, а потребное ΔV:
                    # «Мастодонт» с пятью баками — минимальная по массе
                    # конструкция под задание, и она сама даёт TWR 1.80
                    # при требуемых хоть 1.25, хоть 1.05. Скидка на этом
                    # не сказывается вовсе, а связка при таком блоке —
                    # тридцать тонн балласта.
                    relief = 0.60 * policy.booster_level
                    req.min_twr = max(1.05, policy.liftoff_twr - relief)

        log.info("Задача: %s | требуемый dV = %.0f м/с | ступеней запланировано: %d",
                 spec.describe(), budget.total, len(requirements))

        solver = StageSolver(self.catalog, policy)
        solutions: list[StageSolution] = []
        problems: list[str] = []
        warnings: list[str] = []

        # Решаем сверху вниз: полезная нагрузка каждой нижней ступени —
        # это всё, что стоит выше неё.
        accumulated = payload_mass
        atmospheric = get_body(spec.home_body).atmosphere_height > 0
        for req in sorted(requirements, key=lambda r: -r.index):
            # Нижняя ступень работает в плотном воздухе — ей нужно чем
            # рулить. Верхним качание желательно, но не обязательно.
            solution = solver.solve(req, accumulated, need_decoupler=req.index > 0,
                                    require_gimbal=(atmospheric and req.index == 0))
            if solution is None:
                problems.append(
                    f"Ступень {req.index} ({req.role}): в каталоге нет сочетания, "
                    f"дающего dV {req.dv_required:.0f} м/с при TWR ≥ {req.min_twr:.2f}")
                continue
            solutions.append(solution)
            accumulated = solution.wet_mass

        solutions.sort(key=lambda s: s.requirement.index)

        boosters = (self.choose_boosters(spec, policy, solutions[0])
                    if solutions else None)

        if solutions:
            total_dv = sum(s.delta_v() for s in solutions)
            if total_dv < budget.total:
                problems.append(f"Суммарный dV {total_dv:.0f} м/с меньше требуемых "
                                f"{budget.total:.0f} м/с")
            # Проверяем тяговооружённость связки целиком: с ускорителями
            # стартует не центральный блок, а всё, что стоит на столе.
            launch_twr = boosters.twr if boosters else solutions[0].twr()
            if launch_twr < 1.15:
                problems.append(f"Стартовый TWR {launch_twr:.2f} < 1.15 — "
                                f"ракета не оторвётся от стола")
            elif launch_twr > 2.4:
                warnings.append(f"Стартовый TWR {launch_twr:.2f} избыточен — "
                                f"лишние потери на сопротивление воздуха")
            for s in solutions[1:]:
                if s.burn_time() > 400:
                    warnings.append(f"Ступень {s.requirement.index}: длинный ожог "
                                    f"{s.burn_time():.0f} с — узел придётся дробить")
        else:
            problems.append("Ни одна ступень не решена — проект пуст")

        blueprint = self._build_blueprint(spec, payload, solutions, boosters)
        design = VehicleDesign(spec=spec, policy=policy, budget=budget,
                               payload=payload, stages=solutions,
                               blueprint=blueprint, problems=problems,
                               warnings=warnings, boosters=boosters)
        if design.viable:
            log.info("Проект готов: %.1f т, dV=%.0f м/с, TWR=%.2f, деталей %d",
                     design.total_mass, design.total_delta_v,
                     design.launch_twr, design.part_count)
        else:
            log.warning("Проект нежизнеспособен: %s", "; ".join(problems))
        return design

    # ------------------------------------------------------------------
    def _build_blueprint(self, spec: MissionSpec, payload: list[PayloadChoice],
                         solutions: list[StageSolution],
                         boosters: BoosterPack | None = None) -> Blueprint:
        pod = next((c.part for c in payload if c.role == "командный модуль"), None)
        chute = next((c.part for c in payload
                      if c.role == "парашют" and not c.radial), None)
        nose = next((c.part for c in payload if c.nose), None)
        # Приборы, уехавшие в служебный отсек, снаружи больше не висят.
        service_bay = next((c.part for c in payload if c.service_bay), None)
        fairing = next((c.part for c in payload if c.fairing), None)
        legs = [c.part for c in payload if c.role == self.LEG_ROLE]
        avionics = next((c.part for c in payload if c.avionics), None)
        payload_stack = [c.part for c in payload if c.stack]
        rover_wheels = [c.part for c in payload
                        if c.role == self.ROVER_ROLE_WHEEL]
        bay_payload = [c.part for c in payload if c.bay] if service_bay else []
        radial = [c.part for c in payload if c.radial and not c.fin
                  and not c.clamp and not c.nose and not c.fairing
                  and not c.avionics
                  and c.role != self.ROVER_ROLE_WHEEL
                  and not (service_bay and c.bay)]
        fins = [c.part for c in payload
                if c.fin and c.role != self.UPPER_FIN_ROLE]
        upper_fins = [c.part for c in payload if c.role == self.UPPER_FIN_ROLE]
        # Одноступенчатому аппарату верхнее оперение — лишний груз:
        # сбрасывать нечего, нижнее никуда не денется.
        if len(solutions) < 2:
            upper_fins = []
        clamps = [c.part for c in payload if c.clamp]
        stages = [StageBuild(engine=s.engine, tanks=[s.tank] * s.tank_count,
                             decoupler=s.decoupler, index=s.requirement.index)
                  for s in solutions]
        name = CraftNamer.name_for(spec)
        pack = None
        if boosters is not None:
            pack = BoosterBuild(booster=boosters.part, decoupler=boosters.decoupler,
                                nose=boosters.nose, count=boosters.count)
        # Переходники между ступенями разной толщины. Считаются ЗДЕСЬ, а
        # не при сборке: топливный FL-A151S несёт топливо и меняет и
        # массу, и ΔV, а сборщик о бюджете не знает.
        adapters: dict[int, PartInfo] = {}
        above = (service_bay or pod)
        for stage in sorted(stages, key=lambda s: -s.index):
            if not stage.tanks:
                continue
            top_d = above.hull_diameter
            bottom_d = stage.tanks[0].hull_diameter
            if abs(top_d - bottom_d) > 1e-6:
                found = self.catalog.oriented_adapter(top_d, bottom_d)
                if found is not None:
                    adapters[stage.index] = found
            above = stage.tanks[0]

        return Blueprint(pod=pod, parachute=chute, stages=stages,
                         radial_payload=radial, fins=fins,
                         upper_fins=upper_fins, clamps=clamps,
                         nose=nose, boosters=pack, name=name,
                         service_bay=service_bay, bay_payload=bay_payload,
                         adapters=adapters, fairing=fairing, legs=legs,
                         avionics=avionics, payload_stack=payload_stack,
                         rover_wheels=rover_wheels)


class CraftNamer:
    """Осмысленные имена чертежей — чтобы в VAB было видно, что это за аппарат."""

    @staticmethod
    def name_for(spec: MissionSpec) -> str:
        from ..config import CONFIG
        return CONFIG.craft.craft_name
