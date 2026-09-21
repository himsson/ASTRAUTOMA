"""Среда сборки ракеты для нейросети-конструктора.

Сеть выбирает детали по одной: на каждом шаге она видит числовое
состояние недособранной ракеты и выдаёт индекс детали из палитры либо
токен «закончить». Никаких формул сети не сообщается — она видит только
массы, тяги и удельные импульсы уже поставленных деталей.

Палитра фиксирована по составу и порядку (сортировка по имени), иначе
после обновления игры индексы бы «поехали» и сохранённые веса потеряли
смысл.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import G0
from ..engineer.parts_catalog import PartInfo, get_catalog
from ..logging_setup import get_logger
from .spaces import BUILDER_SLOT_LIMIT, BuilderState

log = get_logger("neural.builder_env")

# Размер палитры: столько деталей максимум видит сеть-конструктор.
PALETTE_SIZE = 24
# Последний индекс действия — «сборка закончена»
BUILDER_ACTION_COUNT = PALETTE_SIZE + 1
FINISH_ACTION = PALETTE_SIZE


@dataclass
class Palette:
    """Фиксированный набор деталей, из которого выбирает сеть."""
    parts: list[PartInfo] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.parts)

    def get(self, index: int) -> PartInfo | None:
        if 0 <= index < len(self.parts):
            return self.parts[index]
        return None

    def describe(self) -> str:
        lines = []
        for i, part in enumerate(self.parts):
            kind = ("двигатель" if part.is_engine else
                    "бак" if part.is_tank else
                    "разделитель" if part.is_decoupler else
                    "ядро" if part.is_command else
                    "парашют" if part.is_parachute else
                    "стабилизатор" if part.category == "Aero" else
                    "прочее")
            lines.append(f"  [{i:2d}] {kind:12s} {part.name:24s} "
                         f"{part.dry_mass:6.3f} т")
        lines.append(f"  [{FINISH_ACTION:2d}] закончить сборку")
        return "\n".join(lines)


def build_palette(catalog=None, size: int = PALETTE_SIZE) -> Palette:
    """Собирает палитру из каталога игры: ядра, баки, двигатели, обвязка."""
    catalog = catalog or get_catalog()
    chosen: list[PartInfo] = []
    seen: set[str] = set()

    def take(candidates, limit: int) -> None:
        for part in candidates[:limit]:
            if part.name not in seen:
                seen.add(part.name)
                chosen.append(part)

    # Диаметры 0.625/1.25/2.5 покрывают весь спектр задач до Муны
    for diameter in (0.625, 1.25, 2.5):
        take(catalog.engines(diameter=diameter), 3)
        take(catalog.tanks(diameter=diameter), 3)
        take(catalog.decouplers(diameter=diameter), 1)
    take(catalog.probe_cores(), 2)
    take(catalog.parachutes(), 1)
    take([p for p in catalog.fins(steerable=False)], 1)
    take(catalog.batteries(), 1)
    take(catalog.solar_panels(), 1)

    chosen.sort(key=lambda p: p.name)          # стабильный порядок индексов
    if len(chosen) > size:
        chosen = chosen[:size]
    while len(chosen) < size and catalog.parts:
        # добиваем палитру повторами, чтобы размер действия не менялся
        chosen.append(chosen[-1] if chosen else next(iter(catalog.parts.values())))
    return Palette(chosen)


# ==========================================================================
@dataclass
class DesignEvaluation:
    """Оценка собранного аппарата — используется как награда."""
    valid: bool
    delta_v: float
    twr: float
    total_mass: float
    part_count: int
    problems: list = field(default_factory=list)
    # Вердикт проектного бюро (только со ступени 2), см. designer_pro
    verdict: object = None

    @property
    def markers(self) -> list:
        return list(getattr(self.verdict, "markers", []) or [])

    def describe(self) -> str:
        if not self.valid:
            return "негодная сборка: " + "; ".join(self.problems)
        text = (f"ΔV {self.delta_v:.0f} м/с, TWR {self.twr:.2f}, "
                f"масса {self.total_mass:.2f} т, деталей {self.part_count}")
        if self.markers:
            text += " | маркеры бюро: " + ", ".join(self.markers)
        return text


class BuilderEnv:
    """Пошаговая сборка. Интерфейс как у gym: reset() / step(action).

    Награда за шаг нулевая, вся награда — в конце эпизода: она считается
    по тому, что реально получилось (ΔV, TWR, масса). Это функция награды,
    а не подсказка сети: веса меняются только градиентом от этой награды.
    """

    def __init__(self, palette: Palette | None = None, catalog=None,
                 goal_altitude: float = 80_000.0, goal_transfer: bool = False,
                 goal_landing: bool = False, stage: int = 1,
                 seed: int | None = None):
        import random as _random
        self._rng = _random.Random(seed)
        self.catalog = catalog or get_catalog()
        self.palette = palette or build_palette(self.catalog)
        self.goal_altitude = goal_altitude
        self.goal_transfer = goal_transfer
        self.goal_landing = goal_landing
        self.chosen: list[PartInfo] = []
        self.state = BuilderState()
        # Проектное бюро (pro_modules/designer_pro.py) подключается только
        # начиная с Тренажёра+: на базовой ступени сеть ещё не умеет
        # выводить аппарат на орбиту, и разговор про мидель ей рано.
        self.stage = int(stage)
        self.designer = self._attach_designer()

    def _attach_designer(self):
        """Бюро подключается мягко: без него среда работает как раньше."""
        try:
            from pro_modules.designer_pro import build_designer
        except ImportError:
            return None
        try:
            return build_designer(self.stage, goal_altitude=self.goal_altitude,
                                  goal_transfer=self.goal_transfer,
                                  goal_landing=self.goal_landing)
        except Exception as exc:
            log.warning("Проектное бюро не подключилось: %s", exc)
            return None

    # ------------------------------------------------------------------
    def sample_goal(self) -> None:
        """Новая задача на эпизод — начиная со ступени 2.

        На первой ступени цель одна и та же (орбита Кербина 80 км): сеть
        и без того решает задачу «что вообще ставить». Со второй цель
        меняется каждый эпизод, и сеть обязана СЧИТЫВАТЬ её со входа, а не
        заучивать один-единственный удачный набор деталей.
        """
        if self.stage < 2:
            self.goal_altitude, self.goal_transfer, self.goal_landing = \
                80_000.0, False, False
            return
        roll = self._rng.random()
        self.goal_altitude = self._rng.choice(
            [75_000.0, 80_000.0, 100_000.0, 150_000.0, 250_000.0])
        self.goal_transfer = roll > 0.55
        self.goal_landing = roll > 0.80

    def reset(self, goal_altitude: float | None = None,
              goal_transfer: bool | None = None,
              goal_landing: bool | None = None) -> list[float]:
        if goal_altitude is None and goal_transfer is None \
                and goal_landing is None:
            self.sample_goal()
        if goal_altitude is not None:
            self.goal_altitude = goal_altitude
        if goal_transfer is not None:
            self.goal_transfer = goal_transfer
        if goal_landing is not None:
            self.goal_landing = goal_landing
        self.chosen = []
        self.state = BuilderState()
        return self.observation()

    def observation(self) -> list[float]:
        return self.state.encode(self.goal_altitude, self.goal_transfer,
                                 self.goal_landing)

    # ------------------------------------------------------------------
    def action_mask(self) -> list[bool]:
        """Что физически можно поставить следующим шагом.

        Это не подсказка «как строить ракету», а запрет на бессмысленные
        действия: нельзя поставить второе ядро управления или закончить
        сборку, в которой нет ни одной детали.
        """
        mask = [True] * BUILDER_ACTION_COUNT
        for index, part in enumerate(self.palette.parts):
            if part.is_command and self.state.has_command:
                mask[index] = False
        # первым шагом обязателен командный модуль: без него аппарат
        # неуправляем в принципе (в KSP это «Нет управления»)
        if not self.state.has_command:
            for index, part in enumerate(self.palette.parts):
                mask[index] = bool(part.is_command)
            mask[FINISH_ACTION] = False
        if self.state.slot_index >= BUILDER_SLOT_LIMIT:
            mask = [False] * BUILDER_ACTION_COUNT
            mask[FINISH_ACTION] = True
        if not any(mask):
            mask[FINISH_ACTION] = True
        return mask

    # ------------------------------------------------------------------
    def step(self, action: int):
        """Возвращает (observation, reward, done, info)."""
        action = int(action)
        if action == FINISH_ACTION or self.state.slot_index >= BUILDER_SLOT_LIMIT:
            evaluation = self.evaluate()
            reward = self.reward_from(evaluation)
            return self.observation(), reward, True, {
                "evaluation": evaluation, "parts": [p.name for p in self.chosen]}

        part = self.palette.get(action)
        if part is None:
            return self.observation(), -1.0, False, {"invalid": True}

        self.chosen.append(part)
        self._absorb(part)
        return self.observation(), 0.0, False, {"part": part.name}

    # ------------------------------------------------------------------
    def _absorb(self, part: PartInfo) -> None:
        densities = self.catalog.densities
        s = self.state
        s.slot_index += 1
        s.part_count += 1
        s.dry_mass += part.dry_mass * 1000.0
        s.total_mass += part.wet_mass(densities) * 1000.0
        s.fuel_units += part.fuel_units()
        if part.is_engine:
            s.thrust_sum += part.max_thrust * 1000.0
            engines = [p for p in self.chosen if p.is_engine]
            s.isp_mean = sum(p.isp_vac for p in engines) / max(1, len(engines))
            s.has_engine = True
        s.has_command = s.has_command or part.is_command
        s.has_tank = s.has_tank or part.is_tank
        s.has_decoupler = s.has_decoupler or part.is_decoupler
        s.has_power = s.has_power or part.resources.get("ElectricCharge", 0) > 0
        s.last_is_engine = part.is_engine
        s.last_is_tank = part.is_tank
        s.last_diameter = part.diameter

    # ------------------------------------------------------------------
    def evaluate(self) -> DesignEvaluation:
        """Что получилось: ΔV и TWR собранной связки."""
        densities = self.catalog.densities
        problems: list[str] = []
        engines = [p for p in self.chosen if p.is_engine]
        tanks = [p for p in self.chosen if p.is_tank]

        if not any(p.is_command for p in self.chosen):
            problems.append("нет командного модуля")
        if not engines:
            problems.append("нет двигателя")
        if not tanks:
            problems.append("нет топливного бака")

        total_mass = sum(p.wet_mass(densities) for p in self.chosen)
        dry_mass = sum(p.dry_mass for p in self.chosen)
        thrust = sum(p.max_thrust for p in engines)
        isp = (sum(e.isp_vac * e.max_thrust for e in engines) / thrust
               if thrust > 0 else 0.0)

        delta_v = 0.0
        if isp > 0 and dry_mass > 0 and total_mass > dry_mass:
            delta_v = isp * G0 * math.log(total_mass / dry_mass)
        gravity = 9.81
        twr = (thrust * 1000.0) / (total_mass * 1000.0 * gravity) if total_mass else 0.0

        # Со ступени 3 обе главные величины считаются честно.
        #
        # ΔV одной кучей — прямая ложь для многоступенчатой ракеты: там
        # верхняя ступень летит УЖЕ БЕЗ нижней, и её отношение масс
        # совершенно другое. Сложение по Циолковскому «всё топливо против
        # всей сухой массы» систематически ЗАНИЖАЕТ результат для хороших
        # многоступенчатых сборок и завышает для одноступенчатых — то есть
        # награда подталкивала сеть ровно не туда.
        #
        # TWR по всей массе — та же беда с другого конца: на старте
        # работает только нижний двигатель, а не сумма всех.
        staged = None
        if self.stage >= 3:
            staged = self._staged_performance(densities)
            if staged is not None:
                delta_v, twr = staged

        if twr < 1.0 and not problems:
            problems.append(f"TWR {twr:.2f} < 1 — не оторвётся")

        if self.stage >= 3:
            problems += self._buildability_problems()

        verdict = None
        if self.designer is not None:
            # Бюро смотрит на ту же сборку своим чек-листом: тяга,
            # устойчивость, разумная достаточность по топливу, иерархия
            # двигателей и форма корпуса по опыту прошлых полётов
            verdict = self.designer.verify(self.chosen, densities,
                                           delta_v=delta_v, twr=twr)
            problems += [item for item in verdict.advice
                         if verdict.critical and item not in problems]
        return DesignEvaluation(valid=not problems, delta_v=delta_v, twr=twr,
                                total_mass=total_mass,
                                part_count=len(self.chosen), problems=problems,
                                verdict=verdict)

    # ------------------------------------------------------------------
    def _staged_performance(self, densities) -> tuple[float, float] | None:
        """ΔV по ступеням и стартовый TWR по нижнему двигателю.

        Ступени нарезаются так же, как это делает `to_blueprint`: каждый
        двигатель забирает свою долю баков. Считаем снизу вверх — полезной
        нагрузкой каждой ступени служит всё, что стоит над ней.
        """
        blueprint = self.to_blueprint()
        if blueprint is None or not blueprint.stages:
            return None
        # Всё, что не входит ни в одну ступень (ядро, приборы, парашют)
        above = sum(p.wet_mass(densities) for p in self.chosen)
        for stage in blueprint.stages:
            above -= sum(t.wet_mass(densities) for t in stage.tanks)
            above -= stage.engine.wet_mass(densities)
            if stage.decoupler is not None:
                above -= stage.decoupler.wet_mass(densities)
        payload = max(0.0, above)

        total_dv = 0.0
        accumulated = payload
        bottom_twr = 0.0
        # Верхние ступени первыми: их масса войдёт в нагрузку нижних
        for index, stage in enumerate(reversed(blueprint.stages)):
            fuel = sum(t.wet_mass(densities) - t.dry_mass for t in stage.tanks)
            dry = (sum(t.dry_mass for t in stage.tanks) + stage.engine.dry_mass
                   + (stage.decoupler.dry_mass if stage.decoupler else 0.0))
            wet = dry + fuel
            isp = stage.engine.isp_vac
            if isp <= 0 or fuel <= 0:
                continue
            start = accumulated + wet
            end = accumulated + dry
            if start > end > 0:
                total_dv += isp * G0 * math.log(start / end)
            accumulated = start
            if index == len(blueprint.stages) - 1:      # это нижняя ступень
                bottom_twr = (stage.engine.max_thrust * 1000.0
                              / (start * 1000.0 * 9.81)) if start > 0 else 0.0
        return total_dv, bottom_twr

    def _buildability_problems(self) -> list[str]:
        """Собирается ли это вообще в .craft и проходит ли валидатор.

        Ступень 3 засчитывает только то, что игра действительно откроет.
        Сборка, красивая по числам, но нестыкуемая по узлам, — не проект.
        """
        from ..engineer.craft_writer import CraftAssembler, validate_craft

        blueprint = self.to_blueprint()
        if blueprint is None:
            return ["чертёж не собирается: нет ядра, двигателя или бака"]
        try:
            text = CraftAssembler(catalog=self.catalog, seed=0).render(blueprint)
        except Exception as exc:
            return [f"сборщик не смог построить чертёж: {exc}"]
        problems = validate_craft(text)
        return [f"валидатор .craft: {p}" for p in problems[:3]]

    def reward_from(self, ev: DesignEvaluation) -> float:
        """Награда за конструкцию (для предобучения без запуска игры).

        Форма награды простая и честная: сколько ΔV удалось получить,
        поднялась ли ракета вообще (TWR) и не стала ли она чрезмерно
        тяжёлой. В реальном цикле эту награду заменяет исход полёта.
        """
        if not ev.valid:
            return -1.0 - 0.1 * len(ev.problems)
        need = 3400.0 + (1000.0 if self.goal_transfer else 0.0) \
            + (1200.0 if self.goal_landing else 0.0)
        dv_score = min(1.5, ev.delta_v / need)
        twr_score = 1.0 - min(1.0, abs(ev.twr - 1.7) / 1.7)
        mass_score = max(0.0, 1.0 - ev.total_mass / 120.0)
        reward = 2.0 * dv_score + 0.7 * twr_score + 0.3 * mass_score

        if self.stage >= 2:
            # Беспилотник без энергии — это «Нет управления» на второй
            # минуте полёта. На первой ступени об этом ещё рано: сеть там
            # учится вообще ставить детали в осмысленном порядке.
            if not any(p.resources.get("ElectricCharge", 0) > 0
                       for p in self.chosen):
                reward -= 0.6
            if not any(p.is_antenna for p in self.chosen):
                reward -= 0.2
        if self.stage >= 3:
            # На третьей ступени за расточительство платят: каждая деталь
            # сверх десяти и каждая тонна сверх двадцати вычитаются. Без
            # этого сеть находит вырожденное решение — навесить всё, что
            # даёт ΔV, и не думать о сборке.
            reward -= 0.03 * max(0, ev.part_count - 10)
            reward -= 0.01 * max(0.0, ev.total_mass - 20.0)

        if self.designer is not None and ev.verdict is not None:
            # Со ступени 2 к награде добавляются штрафы бюро: за овер-
            # инжиниринг по топливу, неустойчивость, неверную иерархию
            # двигателей и игнорирование опыта прошлых полётов
            reward = self.designer.reward(reward, ev.verdict)
        return reward

    # ------------------------------------------------------------------
    def to_blueprint(self):
        """Переводит выбор сети в чертёж для .craft (если он вообще собираем)."""
        from ..engineer.craft_writer import Blueprint, StageBuild

        cores = [p for p in self.chosen if p.is_command]
        engines = [p for p in self.chosen if p.is_engine]
        tanks = [p for p in self.chosen if p.is_stack_tank]
        decouplers = [p for p in self.chosen if p.is_decoupler]
        chutes = [p for p in self.chosen if p.is_parachute]
        radial = [p for p in self.chosen
                  if not (p.is_command or p.is_engine or p.is_stack_tank
                          or p.is_decoupler or p.is_parachute)
                  and p.can_surface_attach]
        if not cores or not engines or not tanks:
            return None

        # Ступени: каждый двигатель забирает свою долю баков
        stages: list[StageBuild] = []
        per_engine = max(1, len(tanks) // len(engines))
        for index, engine in enumerate(engines):
            slice_tanks = tanks[index * per_engine:(index + 1) * per_engine]
            if not slice_tanks:
                slice_tanks = [tanks[-1]]
            decoupler = decouplers[index] if index < len(decouplers) else None
            stages.append(StageBuild(engine=engine, tanks=list(slice_tanks),
                                     decoupler=decoupler, index=index))
        stack_chute = next((c for c in chutes if c.bottom_node), None)
        radial += [c for c in chutes if c.bottom_node is None]
        return Blueprint(pod=cores[0], parachute=stack_chute, stages=stages,
                         radial_payload=radial, name="KIA_Neural")
