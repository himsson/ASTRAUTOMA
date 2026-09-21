"""Генератор .craft файлов KSP (ConfigNode) из чертежа Инженера.

Что было исправлено по сравнению с первой версией (из-за неё в VAB
появлялся только один двигатель):

1. Имена деталей берутся из реальных cfg игры (каталог), а не из таблицы
   в коде: `mk1pod_v2`, `liquidEngine_v2`, `Decoupler_1`, `Rockomax16_BW`...
2. Идентификатор в .craft пишется как `<имя с '_'→'.'>_<uid>` — именно так
   его пишет и разбирает KSP (`liquidEngine3_v2` -> `liquidEngine3.v2_4294166486`).
   Раньше подчёркивания в имени ломали разбор, и KSP выбрасывал деталь.
3. Позиции деталей вычисляются по настоящим стыковочным узлам из cfg
   (`node_stack_top/bottom` × rescaleFactor), поэтому двигатель садится
   ровно на срез бака, а не висит в воздухе.
4. Связи пишутся с обеих сторон: `link` у родителя + `attN = <узел>,<id>_x|y|z`
   у обеих деталей (формат KSP 1.6+), радиальные — через `srfN`.
5. Между ступенями всегда ставится декаплер, стадии нумеруются по правилу
   KSP: первой срабатывает стадия с наибольшим istg, парашют — istg 0.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CONFIG
from ..logging_setup import get_logger
from .config_node import ConfigNode
from .parts_catalog import NODE_SIZE_M, PartInfo, get_catalog

log = get_logger("engineer.craft")

_KSP_VERSION_CACHE: str | None = None


def ksp_version(default: str = "1.12.5") -> str:
    """Читает версию игры из установки — её обязан знать заголовок .craft.

    KSP сверяет поле version при загрузке чертежа и прогоняет старые файлы
    через конвейер обновления, меняя правила разбора полей. Поэтому версия
    определяется по readme.txt самой игры, а не задаётся константой.
    """
    global _KSP_VERSION_CACHE
    if _KSP_VERSION_CACHE:
        return _KSP_VERSION_CACHE
    root = CONFIG.craft.ksp_root
    if root:
        readme = Path(root) / "readme.txt"
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8", errors="replace")[:4000]
                match = re.search(r"Version\s+(\d+\.\d+\.\d+)", text)
                if match:
                    _KSP_VERSION_CACHE = match.group(1)
                    log.info("Версия KSP из установки: %s", _KSP_VERSION_CACHE)
                    return _KSP_VERSION_CACHE
            except OSError:
                pass
    _KSP_VERSION_CACHE = default
    return default


# ==========================================================================
@dataclass
class StageBuild:
    """Одна ступень: бак(и) сверху, двигатель снизу, под ним — декаплер."""
    engine: PartInfo
    tanks: list[PartInfo]
    decoupler: PartInfo | None = None     # отделяет ступень, лежащую НИЖЕ
    index: int = 0                        # 0 — нижняя (стартовая)

    def parts(self) -> list[PartInfo]:
        out = list(self.tanks) + [self.engine]
        if self.decoupler:
            out.append(self.decoupler)
        return out


# На сколько метров отодвинуть пусковую мачту за обшивку. Её собственный
# диаметр каталог не знает (в cfg он не указан), а ставится деталь на
# поверхность корпуса своим ЦЕНТРОМ — то есть наполовину внутрь бака.
# Снимок из игры это и показал: мачты слились с корпусом в один ком.
CLAMP_STANDOFF = 0.30

# Ниже какого смещения узел крепления считается «в центре детали».
# У TT-70 оно −0.03 м — это не геометрия, а погрешность модели, и
# принимать её за направление нельзя.
NODE_OFFSET_EPS = 0.10

# Вылет радиального разделителя за обшивку, метры. Замер по штатным
# чертежам KSP: TT-38K на баке радиусом 0.625 ставит свой центр на 0.644.
DECOUPLER_ARM = 0.22


def outward_axis(info: PartInfo) -> tuple[float, float, float]:
    """В какую сторону в СВОИХ координатах деталь смотрит наружу.

    Это не догадка: правило снято с 87 замеров по штатным чертежам KSP
    (`Ships/VAB/*.craft`), где детали расставлены самой игрой.

    Наружу смотрит сторона, ПРОТИВОПОЛОЖНАЯ смещению узла крепления:
    узел лежит там, где деталь прижимается к корпусу, значит тело детали
    уходит в другую сторону. У пусковой мачты узел на +Z (0.807) — она
    смотрит −Z; у твердотопливного ускорителя узел на −Z (−0.625) — он
    смотрит +Z.

    Если узел стоит в самом центре детали (оперение, батарея), смещение
    ничего не говорит, и остаётся направление узла. Замеры показывают,
    что наружу в этом случае смотрит ОТРИЦАТЕЛЬНАЯ сторона его оси:
    у оперения направление +X, а наружу — −X.
    """
    node = info.surface_node or (0.0, 0.0, 0.0)
    horizontal = math.hypot(node[0], node[2])
    if horizontal > NODE_OFFSET_EPS:
        return (-node[0] / horizontal, 0.0, -node[2] / horizontal)
    orient = info.surface_node_orient or (1.0, 0.0, 0.0)
    if abs(orient[0]) >= abs(orient[2]):
        return (-1.0, 0.0, 0.0)
    return (0.0, 0.0, -1.0)


def clamp_body_offset(clamp: PartInfo) -> float:
    """На сколько метров корпус мачты ниже точки её захвата.

    Берётся из вертикальной составляющей узла крепления: у TT18-A это
    1.354 м. Поворот вокруг вертикали её не меняет, поэтому кватернион
    здесь не нужен.
    """
    node = clamp.surface_node or (0.0, 0.0, 0.0)
    return float(node[1])


def _quat_mul(a, b) -> tuple[float, float, float, float]:
    """Произведение кватернионов (x, y, z, w): сначала b, потом a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def mounts_vertically(info: PartInfo) -> bool:
    """Деталь рассчитана стоять на ГОРИЗОНТАЛЬНОЙ площадке, осью вверх.

    Такую на цилиндрическом борту надо завалить набок, иначе она
    прижмётся к корпусу донцем и ляжет плашмя. Оператор увидел это в
    ангаре: ретранслятор RA-2 обхватил ядро кольцом вместо того, чтобы
    торчать в сторону.

    Признак — направление узла крепления вдоль вертикали. Замер по
    каталогу: у всех антенн это (0, −1, 0), у аккумулятора и датчиков
    (0, 0, −1), у оперения и панелей (1, 0, 0). Замер по чужим чертежам
    подтверждает: `RelayAntenna5` игра ставит с поворотом
    (0.707, 0, 0, 0.707) — ровно девяносто градусов вокруг X.
    """
    orient = info.surface_node_orient
    if not orient:
        return False
    return abs(orient[1]) > max(abs(orient[0]), abs(orient[2]))


def _rotate(quat, vector) -> tuple[float, float, float]:
    """Поворот вектора кватернионом (x, y, z, w)."""
    x, y, z, w = quat
    vx, vy, vz = vector
    return (vx * (1 - 2 * (y * y + z * z)) + vy * 2 * (x * y - z * w)
            + vz * 2 * (x * z + y * w),
            vx * 2 * (x * y + z * w) + vy * (1 - 2 * (x * x + z * z))
            + vz * 2 * (y * z - x * w),
            vx * 2 * (x * z - y * w) + vy * 2 * (y * z + x * w)
            + vz * (1 - 2 * (x * x + y * y)))

# Зазор между бортом центрального блока и бортом ускорителя, метры.
# Ноль ставить нельзя: обшивки соприкоснутся, и KSP засчитает столкновение.
# Значение снято с чертежа `УРНА-3`: ускоритель BACC стоит центром на
# 1.951 при обшивке бака 1.250 и своей 0.625, то есть зазор 0.076 м.
# Берём 0.08 — столько же, сколько у самой игры.
BOOSTER_GAP = 0.08


@dataclass
class BoosterBuild:
    """Связка бокового ускорителя: разделитель + ускоритель + конус.

    Ускорители навешиваются ТОЛЬКО чётным числом и строго симметрично.
    Нечётная или несимметричная навеска даёт боковую составляющую тяги,
    которую нечем парировать: аппарат уводит с курса сразу после отрыва.
    """
    booster: PartInfo
    decoupler: PartInfo | None = None
    nose: PartInfo | None = None
    count: int = 2

    def parts(self) -> list[PartInfo]:
        one = [self.booster]
        if self.decoupler:
            one.append(self.decoupler)
        if self.nose:
            one.append(self.nose)
        return one * self.count


@dataclass
class Blueprint:
    """Полный чертёж: полезная нагрузка + ступени (stages[0] — нижняя)."""
    pod: PartInfo
    parachute: PartInfo | None = None
    stages: list[StageBuild] = field(default_factory=list)
    radial_payload: list[PartInfo] = field(default_factory=list)
    fins: list[PartInfo] = field(default_factory=list)
    # Оперение ВЕРХНЕЙ ступени. Ставится отдельно от нижнего, потому что
    # нижнее уходит вместе с первой ступенью, а разделение происходит
    # глубоко в воздухе. Замер живого полёта (запуск #127):
    #
    #     t=65.1  напор 14149 Па  тангаж 46.3  крен   0.6  AoA  -6.5
    #     t=65.9  РАЗДЕЛЕНИЕ      тангаж 32.9  крен  -0.6  AoA -19.6
    #     t=66.7  напор 10669 Па  тангаж -8.7  крен -172   AoA  42.6
    #
    # Полторы секунды от ровного полёта до полного переворота: у второй
    # ступени не остаётся ни одной поверхности, которая держала бы её
    # по потоку. Маховик (5 кН·м) против аэродинамического момента на
    # растущем угле не тянет.
    upper_fins: list[PartInfo] = field(default_factory=list)
    # Носовой обтекатель на макушку: в стоковой аэродинамике KSP открытый
    # торец наверху набирает полное сопротивление и съедает сотни м/с.
    nose: PartInfo | None = None
    # Боковые ускорители — то, чем тяжёлая ракета отрывается от стола,
    # не утаскивая на орбиту центральный блок вдвое большего размера.
    boosters: BoosterBuild | None = None
    # Пусковые мачты: держат аппарат на столе до зажигания и
    # отстреливаются вместе с ним. Без них высокая тонкая ракета в KSP
    # заваливается ещё до старта — отказ, которого тренажёр не знает.
    clamps: list[PartInfo] = field(default_factory=list)
    # Служебный отсек и то, что спрятано ВНУТРИ него. Приборы, навешанные
    # снаружи, в стоковой аэродинамике KSP каждый набирает собственное
    # лобовое сопротивление и нагрев; внутри закрытой оболочки они не
    # обдуваются вовсе.
    service_bay: PartInfo | None = None
    bay_payload: list[PartInfo] = field(default_factory=list)
    # Основание защитного обтекателя. Оболочка не деталь, а набор сечений
    # внутри модуля — синтезируется по высоте и радиусу того, что она
    # укрывает (см. `_fairing_sections`).
    fairing: PartInfo | None = None
    # Посадочные опоры. Ставятся на низ ПОСАДОЧНОЙ (верхней) ступени —
    # той, что реально касается грунта, а не на стартовый блок.
    legs: list[PartInfo] = field(default_factory=list)
    # Маховик. Стоит в стеке под ядром: он не навесной, а проходной.
    avionics: PartInfo | None = None
    # КОРПУС АППАРАТА: стековые детали между ядром и ступенями.
    #
    # Спутник связи собирается не как ракета: батареи Z-1k нельзя
    # прицепить сбоку — они стоят В корпусе, одна на другой, как на
    # эталоне оператора. Отсюда отдельный список: он ложится в колонну
    # сразу под ядром и маховиком.
    payload_stack: list[PartInfo] = field(default_factory=list)
    # Колёса лунохода: им нужна своя расстановка, венцом они не ставятся.
    rover_wheels: list[PartInfo] = field(default_factory=list)
    # Переходники между ступенями разной толщины: ключ — индекс ступени,
    # переходник ставится НАД её первым баком. Выбираются конструктором,
    # поэтому входят и в массу, и в число деталей, и в бюджет ΔV.
    adapters: dict[int, PartInfo] = field(default_factory=dict)
    name: str = "KIA_Autogen"

    def all_parts(self) -> list[PartInfo]:
        out = [self.pod]
        if self.parachute:
            out.append(self.parachute)
        out.extend(self.radial_payload)
        out.extend(self.fins)
        out.extend(self.upper_fins)
        out.extend(self.clamps)
        if self.service_bay:
            out.append(self.service_bay)
        out.extend(self.bay_payload)
        out.extend(self.adapters.values())
        if self.fairing:
            out.append(self.fairing)
        out.extend(self.legs)
        if self.avionics:
            out.append(self.avionics)
        out.extend(self.payload_stack)
        out.extend(self.rover_wheels)
        if self.nose:
            out.append(self.nose)
        if self.boosters:
            out.extend(self.boosters.parts())
        for stage in self.stages:
            out.extend(stage.parts())
        return out

    @property
    def part_count(self) -> int:
        return len(self.all_parts())


# ==========================================================================
@dataclass
class PlacedPart:
    info: PartInfo
    uid: int
    index: int
    position: tuple[float, float, float]
    stage: int                      # istg
    decouple_stage: int             # dstg
    stage_icon_order: int = -1      # sqor
    stage_icon_index: int = -1      # sidx
    parent: "PlacedPart | None" = None
    parent_node: str = ""           # узел родителя, к которому прицеплены
    own_node: str = ""              # собственный узел
    surface_attached: bool = False
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    children: list["PlacedPart"] = field(default_factory=list)

    @property
    def craft_id(self) -> str:
        return f"{self.info.craft_id_name}_{self.uid}"

    @property
    def attach_offset(self) -> tuple[float, float, float]:
        if self.parent is None:
            return self.position
        return tuple(a - b for a, b in zip(self.position, self.parent.position))


class CraftAssembler:
    """Собирает .craft-файл из чертежа."""

    def __init__(self, craft_name: str | None = None, catalog=None,
                 seed: int | None = None):
        self.craft_name = craft_name or CONFIG.craft.craft_name
        self.catalog = catalog or get_catalog()
        self._rng = random.Random(seed)
        self._uid = 4_294_900_000 - self._rng.randint(0, 200_000)
        self.placed: list[PlacedPart] = []

    def _next_uid(self) -> int:
        self._uid -= self._rng.randint(37, 991)
        return self._uid

    # ------------------------------------------------------------------
    # Сборка дерева деталей
    # ------------------------------------------------------------------
    def assemble(self, bp: Blueprint) -> list[PlacedPart]:
        """Строит дерево от командного модуля вниз. Корень = pod (индекс 0)."""
        self.placed = []
        self._fairing_base = None
        self._fairing_sections = []
        n_stages = len(bp.stages)
        # Нумерация стадий KSP: первой срабатывает наибольшая.
        # stage i (0 = нижняя): двигатель istg = 2*(N-i)-1, декаплер под ним = 2*(N-i)-2
        pod = self._place(bp.pod, position=(0.0, 0.0, 0.0), stage=-1,
                          decouple_stage=0, parent=None)

        # Парашют — сверху на командный модуль, срабатывает последним (стадия 0)
        top, top_node = pod, "top"
        if bp.parachute is not None:
            chute = self._attach_stack(bp.parachute, parent=pod, parent_node="top",
                                       own_node="bottom", stage=0, decouple_stage=0,
                                       icon=0)
            # Стековый парашют занял верхний узел пода — конус пойдёт на него.
            # Радиальный парашют висит на борту и макушку не трогает.
            if not bp.parachute.can_surface_attach or bp.parachute.bottom_node:
                top, top_node = chute, "top"

        # Носовой обтекатель — самая верхняя деталь. Своей стадии у него
        # нет: он не отделяется, а летит с аппаратом до конца.
        if bp.nose is not None and top.info.node(top_node) is not None:
            self._attach_stack(bp.nose, parent=top, parent_node=top_node,
                               own_node="bottom", stage=-1, decouple_stage=0)

        # Служебный отсек — сразу под ядром, до баков. Приборы уезжают
        # внутрь него, и снаружи на корпусе не остаётся ничего лишнего.
        current = pod
        current_node = "bottom"
        # Маховик — сразу под ядром, до отсека: короткий путь к центру
        # масс, и он не мешает приборам внутри отсека.
        if bp.avionics is not None:
            wheel = self._attach_stack(bp.avionics, parent=current,
                                       parent_node=current_node,
                                       own_node="top", stage=-1,
                                       decouple_stage=0)
            current, current_node = wheel, "bottom"
        # Корпус аппарата: батареи и прочее, что стоит В колонне.
        for part in bp.payload_stack:
            block = self._attach_stack(part, parent=current,
                                       parent_node=current_node,
                                       own_node="top", stage=-1,
                                       decouple_stage=0)
            current, current_node = block, "bottom"
        if bp.service_bay is not None:
            bay = self._attach_stack(bp.service_bay, parent=current,
                                     parent_node=current_node, own_node="top",
                                     stage=-1, decouple_stage=0)
            current, current_node = bay, "bottom"
            # ПРИБОРЫ СТАВИМ БЛИЖЕ К ОСИ, А НЕ К СТЕНКЕ.
            #
            # У стенки они вылезали наружу: сама деталь занимает место от
            # своего начала координат в сторону обшивки, и запаса в
            # четверть метра не хватало — на снимке из ангара аккумулятор
            # и датчик торчали сквозь корпус. Доля от радиуса надёжнее
            # фиксированного отступа: она работает и в узком отсеке.
            inner = max(0.05, bay.info.hull_diameter * self.BAY_INNER_FRACTION)
            for idx, part in enumerate(bp.bay_payload):
                count = max(1, len(bp.bay_payload))
                self._attach_radial(part, host=bay, slot=idx,
                                    angle_deg=(360.0 / count) * idx,
                                    height_fraction=0.0,
                                    radius_override=inner,
                                    y_override=bay.position[1])

        # Ступени сверху вниз: последняя в списке — верхняя
        # Нижний бак КАЖДОЙ ступени — чтобы было куда ставить оперение
        # верхней ступени, а не только нижней.
        stage_last_tank: dict[int, PlacedPart] = {}
        for stage in reversed(bp.stages):
            i = stage.index
            ignite = 2 * (n_stages - i) - 1
            separate = 2 * (n_stages - i) - 2
            for n, tank in enumerate(stage.tanks):
                if n == 0 and bp.adapters.get(i) is not None:
                    # Переходник закрывает уступ между ступенями разной
                    # толщины. Выбран конструктором — значит учтён в
                    # массе, в числе деталей и в бюджете ΔV.
                    current = self._attach_stack(
                        bp.adapters[i], parent=current,
                        parent_node=current_node, own_node="top",
                        stage=-1, decouple_stage=max(0, separate))
                    current_node = "bottom"
                current = self._attach_stack(tank, parent=current,
                                             parent_node=current_node,
                                             own_node="top", stage=-1,
                                             decouple_stage=max(0, separate))
                current_node = "bottom"
                stage_last_tank[i] = current
            # ОБТЕКАТЕЛЬ УКРЫВАЕТ И САМУ ВЕРХНЮЮ СТУПЕНЬ.
            #
            # Её двигатель работает уже в вакууме, а бак с приборами всё
            # выведение торчит в поток. Основание встаёт МЕЖДУ баком и
            # двигателем — так створки растут от самого низа ступени и
            # укрывают её целиком вместе с нагрузкой. Топливу это не
            # мешает: у основания в cfg стоит `fuelCrossFeed = True`.
            #
            # Заодно обтекатель становится длинным, а длинному идёт и
            # больший диаметр — иначе он выглядит карандашом.
            if bp.fairing is not None and i == max(s.index for s in bp.stages):
                # СТВОРКАМ НУЖНА СВОЯ СТАДИЯ, ИНАЧЕ ОНИ НЕ РАСКРОЮТСЯ.
                #
                # Без неё обтекатель летит до конца миссии закрытым: груз
                # под ним не работает, а масса тащится на орбиту. В
                # чертеже из поставки игры у основания стоит `istg = 2` —
                # то есть сброс приурочен к отделению ступени.
                #
                # Сбрасываем вместе с той ступенью, что лежит ПОД верхней:
                # к этому моменту плотная атмосфера уже позади.
                # СТВОРКИ СБРАСЫВАЕТ ПИЛОТ, А НЕ СТАДИЯ.
                #
                # Стадия для обтекателя выглядела разумно, но привязывала
                # сброс к выгоранию двигателя, а не к выходу из атмосферы.
                # В живом полёте створки так и не раскрылись: панели под
                # ними не видели солнца, заряд сел, аппарат закрутило. И
                # вдобавок пустая стадия ела шаг автостейджинга — в
                # журнале это «Отделена ступень — доступная тяга 0 кН».
                # Теперь сброс делает пилот сразу за атмосферой
                # (`jettison_fairings`), а стадии у створок нет.
                base = self._attach_stack(bp.fairing, parent=current,
                                          parent_node=current_node,
                                          own_node="top", stage=-1,
                                          decouple_stage=max(0, separate))
                current, current_node = base, "bottom"
                self._fairing_base = base
            current = self._attach_stack(stage.engine, parent=current,
                                         parent_node=current_node, own_node="top",
                                         stage=ignite, decouple_stage=max(0, separate),
                                         icon=ignite)
            current_node = "bottom"
            if stage.decoupler is not None and i > 0:
                # РАЗДЕЛИТЕЛЬ И ДВИГАТЕЛЬ НАД НИМ — В ОДНОЙ СТАДИИ.
                #
                # Так расставляет стадии сама игра. Замер по готовым
                # чертежам из сохранения оператора:
                #
                #     Ариан-5:     разделитель 0,  двигатели [3, 0]
                #     Наполлон:    разделители [10,7,6,2], двигатели [11,10,5,4]
                #     Салли-Хат-1: разделители [2, 0], двигатели [4, 2]
                #     SWM-94:      разделитель 1,  двигатель 1
                #
                # Одно нажатие сбрасывает отработавшую ступень и сразу
                # зажигает следующую.
                #
                # Здесь пробовали оба соседних варианта, и оба плохи.
                # `ignite - 1` (было изначально): двигатель зажигается
                # ПРИКРУЧЕННЫМ к отработавшей ступени и секунду жжёт в
                # неё, а следом стадия 0 выбрасывает разделитель вместе с
                # парашютом — купол раскрывался посреди подъёма.
                # `ignite + 1`: между сбросом и зажиганием две секунды без
                # тяги на плотном воздухе, и аппарат за это время
                # разворачивает потоком (апоапсис падал с 37.7 до 31.8 км,
                # скорость с 728 до 484 м/с).
                #
                # Общая стадия убирает и то, и другое.
                drop_first = ignite
                current = self._attach_stack(stage.decoupler, parent=current,
                                             parent_node=current_node,
                                             own_node="top", stage=drop_first,
                                             decouple_stage=drop_first,
                                             icon=drop_first)
                current_node = "bottom"

        # Радиальная мелочь — на верхний бак (или на под, если баков нет)
        host = self._radial_host(pod)
        for idx, part in enumerate(bp.radial_payload):
            self._attach_radial(part, host=host, slot=idx)

        # КОЛЁСА СТАВЯТСЯ КАК У МАШИНЫ, А НЕ ВЕНЦОМ ВОКРУГ КОРПУСА.
        #
        # Обычная радиальная расстановка делит окружность поровну, и
        # шесть колёс встали звездой вокруг цилиндра. Оператор прислал
        # снимок из ангара с одним словом: «брелятина». Он прав —
        # ездить на таком нельзя: половина колёс смотрит вверх.
        #
        # Машина устроена иначе: два ряда по бортам (90° и 270°), по три
        # колеса в ряд, разнесённые вдоль корпуса. Тогда все шесть стоят
        # на грунте, а корпус между ними — это и есть шасси.
        if bp.rover_wheels:
            per_side = max(1, len(bp.rover_wheels) // 2)
            for idx, wheel in enumerate(bp.rover_wheels):
                side = 90.0 if idx % 2 == 0 else 270.0
                row = idx // 2
                # Ряды раскладываем вдоль корпуса симметрично центру:
                # при трёх рядах это −0.3, 0.0, +0.3 доли высоты.
                offset = (row - (per_side - 1) / 2.0) * 0.3
                self._attach_radial(wheel, host=host, slot=idx,
                                    angle_deg=side,
                                    height_fraction=offset)

        # Стабилизаторы — симметрично на нижний бак, у самого среза
        if bp.fins:
            base = self._lowest_tank()
            if base is not None:
                count = len(bp.fins)
                for idx, fin in enumerate(bp.fins):
                    self._attach_radial(fin, host=base, slot=idx,
                                        angle_deg=(360.0 / count) * idx,
                                        height_fraction=-0.35)

        # Оперение верхней ступени — на её собственный нижний бак. Оно
        # обязано отделяться ВМЕСТЕ с верхней ступенью, а не с первой,
        # поэтому и хозяин у него другой: см. `upper_fins` в чертеже.
        if bp.upper_fins and bp.stages:
            # Не на САМУЮ верхнюю ступень, а на ту, что станет НИЖНЕЙ
            # после первого разделения: именно она остаётся в воздухе на
            # 20 км при напоре 14 кПа. У двухступенчатой это одна и та же
            # ступень, у трёхступенчатой — разные, и разница решает:
            # верхняя к тому времени уже за атмосферой, а средняя нет.
            above_first = min((s.index for s in bp.stages if s.index > 0),
                              default=bp.stages[-1].index)
            base = stage_last_tank.get(above_first)
            if base is not None:
                count = len(bp.upper_fins)
                for idx, fin in enumerate(bp.upper_fins):
                    self._attach_radial(fin, host=base, slot=idx,
                                        angle_deg=(360.0 / count) * idx,
                                        height_fraction=-0.35)

        # Пусковые мачты — тоже симметрично на нижний бак, но ниже
        # стабилизаторов и с ОТДЕЛЬНОЙ стадией: они отстреливаются на
        # зажигании нижнего двигателя. Стадия зажигания нижней ступени
        # (индекс 0) по здешней нумерации — 2*N-1, она же срабатывает
        # первой.
        if bp.clamps:
            base = self._lowest_tank()
            if base is not None:
                release = 2 * n_stages - 1
                count = len(bp.clamps)
                # МАЧТЫ РАЗВОРАЧИВАЕМ И ОТ УСКОРИТЕЛЕЙ ТОЖЕ.
                #
                # Прежний сдвиг уводил их только от стабилизаторов, а про
                # связку ничего не знал. На снимке из ангара видно, чем
                # это кончается: захваты мачт торчат прямо сквозь корпуса
                # ускорителей. Связка навешивается по тем же углам, что и
                # стабилизаторы, поэтому достаточно одного правила —
                # ставить мачты РОВНО МЕЖДУ занятыми направлениями.
                if bp.boosters is not None and bp.boosters.count:
                    # Связка занимает углы `booster_shift + k·360/N`.
                    # Мачты ставим ровно посередине между ними.
                    shift = (self._booster_shift(bp)
                             + 180.0 / bp.boosters.count)
                elif bp.fins:
                    shift = 180.0 / len(bp.fins)
                else:
                    shift = 0.0
                # ВЫСОТА ЗАХВАТА СЧИТАЕТСЯ, А НЕ ЗАДАЁТСЯ ДОЛЕЙ.
                #
                # У TT18-A узел крепления вынесен на 1.354 м ВВЕРХ от её
                # корпуса. Значит корпус всегда оказывается сильно ниже
                # точки захвата, и при доле -0.35 он уходил ниже двигателя:
                #
                #     мачты      y = 2.00   <- нижняя точка всей сборки
                #     двигатель  y = 3.19
                #
                # Игра ставит аппарат так, чтобы нижняя точка касалась
                # стола. Нижней точкой оказывались мачты, ракета спавнилась
                # подвешенной на метр с лишним, при старте физики проседала
                # на захваты и садилась с перекосом. На снимке из игры это
                # видно прямо: аппарат стоит криво ещё до зажигания. Отсюда
                # же и «случайные» опрокидывания сразу после отрыва — они
                # начинались уже наклонёнными.
                #
                # Мачта в KSP сама вытягивается до земли, висеть ниже
                # ракеты ей незачем. Поэтому захват ставится так, чтобы
                # КОРПУС мачты встал вровень с низом аппарата.
                lowest = min((p.position[1] for p in self.placed), default=0.0)
                node = clamp_body_offset(bp.clamps[0])
                grab_y = lowest + node
                for idx, clamp in enumerate(bp.clamps):
                    self._attach_radial(clamp, host=base, slot=idx,
                                        angle_deg=(360.0 / count) * idx + shift,
                                        y_override=grab_y,
                                        stage=release, decouple_stage=release)

        # ПОСАДОЧНЫЕ ОПОРЫ — на низ ВЕРХНЕЙ ступени.
        #
        # Именно она садится: всё, что ниже, сбрасывается по дороге.
        # Опора крепится у самого среза бака, чтобы стойка доставала до
        # грунта, а не висела на полкорпуса выше.
        if bp.legs:
            top_index = max(s.index for s in bp.stages) if bp.stages else 0
            host = stage_last_tank.get(top_index)
            if host is not None:
                foot = host.position[1] - host.info.height / 2.0
                for idx, leg in enumerate(bp.legs):
                    self._attach_radial(leg, host=host, slot=idx,
                                        angle_deg=(360.0 / len(bp.legs)) * idx,
                                        y_override=foot)

        if bp.boosters is not None:
            self._attach_boosters(bp.boosters, n_stages=n_stages,
                                  has_fins=bool(bp.fins))
        self._build_fairing_sections()
        return self.placed

    # Куда обтекатель сводится на макушке, метры радиуса. Ноль ставить
    # нельзя: у створок должна остаться толщина.
    FAIRING_TIP_RADIUS = 0.15
    # Зазор между укрытым грузом и внутренней стенкой створок.
    FAIRING_CLEARANCE = 0.12
    # Запас на размах навесной детали, размеров которой каталог не знает.
    RADIAL_SPAN = 0.45
    # Доля высоты, до которой обтекателю позволено раздаться вширь.
    FAIRING_SLENDERNESS = 0.10

    def _build_fairing_sections(self) -> None:
        """Считает профиль створок по тому, что реально стоит выше основания.

        Сечения задаются парами «высота над основанием — радиус». Берём
        цилиндр до макушки груза и конус над ней: так створки не режут
        полезную нагрузку и не раздуваются впустую.
        """
        base = self._fairing_base
        if base is None:
            return
        covered = [p for p in self.placed
                   if p is not base and p.position[1] > base.position[1]]
        if not covered:
            return
        base_y = base.position[1]
        top = max(p.position[1] + max(p.info.height, 0.2) / 2.0 for p in covered)
        # РАЗМАХ НАВЕСНОЙ ДЕТАЛИ КАТАЛОГ НЕ ЗНАЕТ.
        #
        # У панели, антенны, парашюта нет стыковочных узлов, а значит и
        # `hull_diameter` у них ноль. Пока радиус считался по нему,
        # створки выходили по корпусу отсека — и панели торчали наружу,
        # что оператор и увидел в ангаре. Для навесных берём запас
        # `RADIAL_SPAN`: он покрывает и раскрытую панель.
        widest = 0.0
        for p in covered:
            offset = math.hypot(p.position[0], p.position[2])
            span = (p.info.hull_diameter / 2.0 if not p.surface_attached
                    else self.RADIAL_SPAN)
            widest = max(widest, offset + span)
        height = max(0.6, top - base_y)
        # ДЛИННОМУ ОБТЕКАТЕЛЮ ИДЁТ БОЛЬШИЙ ДИАМЕТР.
        # Пропорция снята с чужих чертежей: там при высоте 8…24 м радиус
        # держится около 0.63…0.66 на основании 1.25 м, то есть примерно
        # десятая часть высоты и не меньше радиуса основания.
        radius = max(base.info.hull_diameter / 2.0,
                     widest + self.FAIRING_CLEARANCE,
                     height * self.FAIRING_SLENDERNESS)
        self._fairing_sections = [
            (0.0, radius),
            (height * 0.62, radius),
            (height, self.FAIRING_TIP_RADIUS),
        ]
        log.info("Обтекатель: створки радиусом %.3f м на высоту %.3f м",
                 radius, height)

    # ------------------------------------------------------------------
    # На сколько градусов связка отвёрнута от стабилизаторов.
    BOOSTER_FIN_SHIFT = 45.0

    def _booster_shift(self, bp: "Blueprint") -> float:
        """Угол, с которого начинается связка. Нужен и мачтам: они
        обязаны встать МЕЖДУ ускорителями, а не в них."""
        return self.BOOSTER_FIN_SHIFT if bp.fins else 0.0

    def _attach_boosters(self, pack: BoosterBuild, n_stages: int,
                         has_fins: bool) -> None:
        """Симметричная навеска боковых ускорителей на центральный блок.

        Стадии. Ускорители зажигаются ВМЕСТЕ с нижним двигателем
        (istg = 2N-1, та же стадия, что и отстрел мачт), а сбрасываются
        сразу следующей стадией 2N-2. Эта стадия свободна: нумерация
        отдаёт 2N-2 разделителю нижней ступени, но у нижней ступени
        разделителя нет — под ней стартовый стол.

        Порядок именно такой, потому что твердотопливный ускоритель
        нельзя выключить: он горит до конца, и висеть на борту мёртвым
        грузом ему нельзя ни секунды лишней.
        """
        core = self._lowest_tank()
        if core is None or pack.count <= 0:
            return
        ignite = 2 * n_stages - 1
        drop = max(0, 2 * n_stages - 2)

        # ЭТАЛОН СНЯТ С ЧУЖОГО ЧЕРТЕЖА, А НЕ ВЫВЕДЕН ИЗ ГОЛОВЫ.
        #
        # Замер по `УРНА-3` (бак Rockomax64, обшивка радиусом 1.250):
        #
        #     радиальный разделитель  ось->центр = 1.254  (на обшивке)
        #     ускоритель BACC         ось->центр = 1.951
        #     зазор между обшивками   = 1.951 − 0.625 − 1.250 = 0.076 м
        #
        # То есть игра ставит связку почти вплотную, в семи сантиметрах.
        # У нас центр ускорителя уезжал на 2.377 при должных 1.741 по двум
        # причинам сразу: радиус считался по УЗЛОВОМУ диаметру бака (2.5
        # вместо обшивки 1.875), и не учитывалось, что `_attach_radial`
        # дополнительно отодвигает начало координат на длину узла. Связка
        # висела в полуметре от корпуса на коротком разделителе — её рвало
        # на старте и уводило аппарат вбок.
        core_r = max(0.35, core.info.hull_diameter / 2.0)
        booster_r = max(0.25, pack.booster.hull_diameter / 2.0)
        node = pack.booster.surface_node or (0.0, 0.0, 0.0)
        node_out = math.hypot(node[0], node[2])
        # Куда должен встать ЦЕНТР ускорителя...
        center = core_r + booster_r + BOOSTER_GAP
        # ...и какую точку крепления для этого задать сборщику: он сам
        # отодвинет начало координат на длину узла.
        stand = max(core_r, center - node_out)
        # Низ ускорителя вровень с низом центрального бака: так тяга
        # приложена симметрично, а сопла не упираются в стартовый стол.
        core_bottom = core.position[1] - core.info.height / 2.0
        y_center = core_bottom + pack.booster.height / 2.0

        # Разворачиваем между стабилизаторами, чтобы не пересечься с ними.
        shift = self.BOOSTER_FIN_SHIFT if has_fins else 0.0
        for idx in range(pack.count):
            angle = (360.0 / pack.count) * idx + shift
            host = core
            if pack.decoupler is not None:
                host = self._attach_radial(
                    pack.decoupler, host=core, slot=idx, angle_deg=angle,
                    stage=drop, decouple_stage=drop,
                    radius_override=core_r, y_override=y_center)
            booster = self._attach_radial(
                pack.booster, host=host, slot=idx, angle_deg=angle,
                stage=ignite, decouple_stage=drop,
                radius_override=stand, y_override=y_center)

            # ВТОРЫМ РАЗДЕЛИТЕЛЕМ УСКОРИТЕЛЬ НЕ ЗАКРЕПИТЬ — ПРОВЕРЕНО.
            #
            # Качание связки — беда настоящая: ускоритель держится
            # серединой, а тяга приложена снизу. Напрашивался второй
            # разделитель у макушки, и я его поставил. На снимке из игры
            # он повис в воздухе: в KSP деталь имеет РОВНО ОДНОГО
            # родителя, и вторая точка крепления физически ничего не
            # держит — просто торчит из корпуса.
            #
            # Правильное средство — расчалки (`strutConnector`): они
            # задаются парой «откуда-куда» и связывают уже стоящие
            # детали. Это отдельная работа по сборщику.
            if pack.nose is not None and pack.booster.top_node is not None:
                self._attach_stack(pack.nose, parent=booster,
                                   parent_node="top", own_node="bottom",
                                   stage=-1, decouple_stage=drop)

    # ------------------------------------------------------------------
    def _place(self, info: PartInfo, position, stage: int, decouple_stage: int,
               parent, parent_node: str = "", own_node: str = "",
               surface: bool = False, rotation=(0.0, 0.0, 0.0, 1.0),
               icon: int = -1) -> PlacedPart:
        part = PlacedPart(
            info=info, uid=self._next_uid(), index=len(self.placed),
            position=position, stage=stage, decouple_stage=decouple_stage,
            stage_icon_order=icon, stage_icon_index=0 if icon >= 0 else -1,
            parent=parent, parent_node=parent_node, own_node=own_node,
            surface_attached=surface, rotation=rotation)
        self.placed.append(part)
        if parent is not None:
            parent.children.append(part)
        return part

    @staticmethod
    def _node_fallback(info: PartInfo, node_name: str,
                       default_top: bool) -> tuple[float, float, float]:
        """Где у детали узел, когда каталог его не отдал.

        Верхний узел — на пол-высоты вверх, нижний — на пол-высоты вниз.
        `default_top` решает спор только для узлов с непонятным именем.
        """
        name = (node_name or "").lower()
        if "top" in name or "верх" in name:
            up = True
        elif "bottom" in name or "низ" in name:
            up = False
        else:
            up = default_top
        half = info.height / 2
        return (0.0, half if up else -half, 0.0)

    def _attach_stack(self, info: PartInfo, parent: PlacedPart, parent_node: str,
                      own_node: str, stage: int, decouple_stage: int,
                      icon: int = -1) -> PlacedPart:
        """Стыковка узел-в-узел: смещаем деталь так, чтобы узлы совпали."""
        p_node = parent.info.node(parent_node)
        c_node = info.node(own_node)
        # Запасная оценка нужна не всем: узлы есть у 270 деталей из 409.
        # Но там, где её применяют, знак обязан зависеть от имени узла.
        # Пока она считала любой узел верхним, обтекатель — единственная
        # деталь, которая цепляется своим НИЖНИМ узлом, — уезжал внутрь
        # корпуса на пол-высоты вместо макушки (замер по чертежу: ядро
        # y=8.196, конус y=8.133 при верных 8.633).
        p_offset = p_node.position if p_node else self._node_fallback(
            parent.info, parent_node, default_top=False)
        c_offset = c_node.position if c_node else self._node_fallback(
            info, own_node, default_top=True)
        position = tuple(parent.position[k] + p_offset[k] - c_offset[k] for k in range(3))
        return self._place(info, position, stage, decouple_stage, parent,
                           parent_node=parent_node, own_node=own_node, icon=icon)

    def _attach_radial(self, info: PartInfo, host: PlacedPart, slot: int,
                       angle_deg: float | None = None,
                       height_fraction: float = 0.25,
                       stage: int = -1,
                       decouple_stage: int | None = None,
                       outward: float = 0.0,
                       radius_override: float | None = None,
                       y_override: float | None = None,
                       origin_on_skin: bool = False) -> PlacedPart:
        """Поверхностная стыковка по окружности несущей детали.

        stage=-1 означает «своей стадии нет» — так крепятся стабилизаторы
        и мелочь на борту. Пусковым мачтам стадия нужна: они
        отстреливаются на зажигании.
        """
        angle = math.radians(angle_deg if angle_deg is not None
                             else (slot * 51.4) % 360.0)
        # `outward` отодвигает деталь ЗА обшивку. Нужно для толстых
        # деталей: генератор ставит на поверхность корпуса их ЦЕНТР, и
        # половина уходит внутрь. У тонкого оперения это незаметно, а
        # пусковая мачта так наполовину влезла в бак — видно на снимке из
        # игры. Собственного диаметра у неё в каталоге нет (0.0), поэтому
        # запас задаётся явно.
        # `radius_override`/`y_override` нужны боковым ускорителям: их
        # выносят не от борта несущей детали, а от оси всей ракеты — иначе
        # ускоритель, подвешенный на тонком разделителе, считает радиус от
        # разделителя и оказывается внутри центрального бака.
        # Радиус берём по ОБШИВКЕ несущей детали, а не по её узлу: у
        # баков размера 1p5 узел объявлен на 2.5 м, а корпус — 1.875 м,
        # и оперение висело в воздухе в 31 см от борта.
        radius = (max(0.35, host.info.hull_diameter / 2.0) + max(0.0, outward)
                  if radius_override is None else radius_override)
        y = (host.position[1] + host.info.height * height_fraction
             if y_override is None else y_override)
        # Радиус выше — это точка НА ОБШИВКЕ, куда придёт узел крепления
        # детали. Куда при этом встанет её начало координат, считается
        # ниже: у мачты узел вынесен на 0.8 м от корпуса, и если посадить
        # туда начало координат, деталь окажется внутри бака. Ровно это и
        # было видно на снимке из игры.
        # Начало отсчёта. Обычная радиальная мелочь меряется от борта своей
        # несущей детали; ускоритель — от ОСИ ракеты (x=z=0), потому что
        # его несущая деталь, разделитель, сама уже вынесена вбок.
        ox, oz = ((host.position[0], host.position[2]) if radius_override is None
                  else (0.0, 0.0))
        attach = (ox + radius * math.cos(angle), y,
                  oz + radius * math.sin(angle))

        # Поворот. Деталь надо развернуть так, чтобы её собственная
        # «наружная» сторона смотрела по радиусу. У разных деталей эта
        # сторона разная (см. outward_axis), и раньше здесь считалось, что
        # у всех она одна и та же, −X: мачты вставали боком, их захваты
        # тянулись мимо ракеты.
        if mounts_vertically(info):
            # Сначала заваливаем деталь набок (90° вокруг X), потом
            # доворачиваем вокруг вертикали так, чтобы её донце смотрело
            # в ось ракеты. После завала направление крепления −Y
            # переходит в −Z, и нужный доворот равен π/2 − φ.
            tip = (math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0))
            psi = math.pi / 2.0 - angle
            yaw = (0.0, math.sin(psi / 2.0), 0.0, math.cos(psi / 2.0))
            rotation = _quat_mul(yaw, tip)
        else:
            out = outward_axis(info)
            own = math.atan2(out[2], out[0])
            # Кватернион с half = -a/2 переводит азимут φ в φ + a, поэтому
            # нужный поворот равен разнице между целевым и собственным.
            half = -(angle - own) / 2.0
            rotation = (0.0, math.sin(half), 0.0, math.cos(half))

        # Начало координат детали: от точки крепления отступаем на её
        # собственное смещение узла, повёрнутое вместе с деталью. Тогда
        # узел ложится ровно на обшивку — так это и записано в штатных
        # чертежах KSP.
        # ОПЕРЕНИЕ — ИСКЛЮЧЕНИЕ, И ЭТО СНЯТО С ЧУЖИХ ЧЕРТЕЖЕЙ.
        #
        # У мачты узел вынесен от корпуса, и сажать на обшивку надо
        # именно узел, иначе деталь влезает в бак. У крылышка узел
        # вынесен ВБОК (`winglet`: 0.638 по X), и то же правило
        # выталкивает деталь наружу — на снимке из игры между корнем
        # крыла и корпусом виден зазор.
        #
        # Замер по `Ships/VAB/*.craft` из сохранения оператора: игрок
        # ставит оперение началом координат ровно на обшивку. Стабилизатор
        # `delta.small` на ускорителе `Clydesdale` (радиус 1.250):
        #
        #     r = 1.237,  1.234,  1.248   (девять замеров)
        #
        # У нас на баке того же радиуса выходило 1.888 — ровно на длину
        # вынесенного узла мимо.
        node = ((0.0, 0.0, 0.0) if origin_on_skin
                else (info.surface_node or (0.0, 0.0, 0.0)))
        shift = _rotate(rotation, node)
        position = (attach[0] - shift[0], attach[1] - shift[1],
                    attach[2] - shift[2])
        return self._place(info, position, stage=stage,
                           decouple_stage=(host.decouple_stage
                                           if decouple_stage is None
                                           else decouple_stage),
                           parent=host, surface=True, rotation=rotation,
                           icon=stage)

    def _radial_host(self, pod: PlacedPart) -> PlacedPart:
        """Куда вешать радиальную мелочь: панели, парашют.

        ПОД СТВОРКИ, А НЕ ПОД ПОТОК. Когда есть служебный отсек, мелочь
        вешается на него: отсек стоит ВЫШЕ основания обтекателя, значит
        всё это оказывается укрыто створками. На верхнем баке она
        оставалась снаружи — а ради того обтекатель и ставится.
        """
        for part in self.placed:
            if part.info.is_cargo_bay:
                return part
        for part in self.placed:
            if part.info.is_stack_tank:
                return part
        return pod

    def _lowest_tank(self) -> PlacedPart | None:
        tanks = [p for p in self.placed if p.info.is_stack_tank]
        return min(tanks, key=lambda p: p.position[1]) if tanks else None

    # ------------------------------------------------------------------
    # Проверка геометрии
    # ------------------------------------------------------------------
    # Насколько деталь вправе не совпасть с бортом несущей, метры. Ноль
    # ставить нельзя: у моделей есть скругления и допуски.
    # Доля диаметра отсека, на которой ставятся приборы внутри него.
    # 0.12 от 1.25 м — это 0.15 м от оси при стенке на 0.625.
    BAY_INNER_FRACTION = 0.12
    RADIAL_TOLERANCE = 0.15
    # Наименьший допустимый зазор между обшивками навесного блока и
    # корпуса. Порог взят чуть мягче эталона: сама игра ставит связку с
    # зазором 0.076 м (замер по `УРНА-3`), и требовать больше — значит
    # ругаться на правильную сборку.
    MIN_SKIN_CLEARANCE = 0.05
    # Насколько стековая деталь вправе разойтись с расчётным стыком.
    STACK_TOLERANCE = 0.05

    def check_geometry(self) -> list[str]:
        """Ищет детали, висящие в воздухе или утопленные в корпус.

        ЭТО ПРОВЕРКА ЦЕЛОГО КЛАССА ОШИБОК, А НЕ ОДНОГО СЛУЧАЯ.
        За сессию их нашлось три, и все — глазами оператора по снимкам
        из игры, а не расчётом:

            обтекатель  — внутри корпуса (узел звался `bottom01`)
            оперение    — в 64 см от борта (узел вынесен вбок)
            оперение    — в 31 см от борта (диаметр брался по узлу,
                          а не по обшивке)

        Каждый раз причина была своя, но след одинаковый: расстояние от
        детали до несущей не совпадает с её геометрией. Такое ловится
        счётом, и теперь ловится ДО полёта.
        """
        problems: list[str] = []
        # ЗАЗОР МЕЖДУ ОБШИВКАМИ НАВЕСНЫХ БЛОКОВ И КОРПУСОМ.
        #
        # Восьми миллиметров хватило, чтобы связка ускорителей взорвалась
        # на старте. Считаем прямо: расстояние от оси минус полуширина
        # ускорителя минус полуширина центрального блока.
        core = self._lowest_tank()
        if core is not None:
            core_r = core.info.hull_diameter / 2.0
            for part in self.placed:
                if "SolidFuel" not in part.info.resources:
                    continue
                axis = math.hypot(part.position[0], part.position[2])
                if axis <= 0.01:
                    continue
                clear = axis - part.info.hull_diameter / 2.0 - core_r
                if clear < self.MIN_SKIN_CLEARANCE:
                    problems.append(
                        f"{part.info.name}: зазор до корпуса {clear:.3f} м "
                        f"(нужно не меньше {self.MIN_SKIN_CLEARANCE:.2f})")
        for part in self.placed:
            host = part.parent
            if host is None:
                continue
            dx = part.position[0] - host.position[0]
            dz = part.position[2] - host.position[2]
            radial = math.hypot(dx, dz)
            if part.surface_attached:
                # Внутри служебного отсека деталь и ДОЛЖНА быть утоплена —
                # ради этого отсек и ставится.
                if host.info.is_cargo_bay:
                    continue
                # Связка ускорителей висит на радиальном разделителе, и её
                # положение задаётся отдельным правилом (зазор до КОРПУСА,
                # проверенный выше). Мерить её от крошечного разделителя
                # бессмысленно: проверка ругалась на верную сборку.
                if host.info.is_decoupler:
                    continue
                skin = max(0.35, host.info.hull_diameter / 2.0)
                # ДЕТАЛЬ СТОИТ НЕ НАЧАЛОМ КООРДИНАТ НА ОБШИВКЕ, А УЗЛОМ.
                #
                # У крылышка узел вынесен вбок на 0.638 м, поэтому его
                # начало координат законно лежит в 0.638 м ЗА бортом.
                # Эталон из чужого чертежа: `R8winglet` на баке радиусом
                # 1.250 стоит на 1.749 — ровно плюс своя длина узла 0.5.
                # Проверка обязана знать это, иначе она ругается на
                # правильную сборку.
                node = part.info.surface_node or (0.0, 0.0, 0.0)
                standoff = math.hypot(node[0], node[2])
                skin_expected = skin + standoff
                # Мачты и ускорители выносят намеренно — у них свой запас.
                allowed = skin_expected + CLAMP_STANDOFF + clamp_body_offset(part.info)
                if radial < skin_expected - self.RADIAL_TOLERANCE:
                    problems.append(
                        f"{part.info.name}: утоплена в {host.info.name} "
                        f"({radial:.3f} при ожидаемых {skin_expected:.3f})")
                elif radial > allowed + self.RADIAL_TOLERANCE:
                    problems.append(
                        f"{part.info.name}: висит в воздухе у {host.info.name} "
                        f"({radial:.3f} при ожидаемых {skin_expected:.3f})")
            else:
                # Стековая стыковка: деталь обязана стоять СООСНО.
                if radial > self.STACK_TOLERANCE:
                    problems.append(
                        f"{part.info.name}: сбита с оси на {radial:.3f} м "
                        f"относительно {host.info.name}")
        return problems

    # ------------------------------------------------------------------
    # Рендеринг ConfigNode
    # ------------------------------------------------------------------
    def render(self, bp: Blueprint) -> str:
        placed = self.assemble(bp)
        # Геометрию проверяем ДО записи чертежа: дешевле поймать здесь,
        # чем увидеть на снимке из игры после сорванного полёта.
        for problem in self.check_geometry():
            log.warning("ГЕОМЕТРИЯ: %s", problem)

        # Сдвигаем сборку так, чтобы нижняя деталь стояла над нулём VAB
        lowest = min((p.position[1] for p in placed), default=0.0)
        shift = 2.0 - lowest
        for part in placed:
            part.position = (part.position[0], part.position[1] + shift,
                             part.position[2])
        heights = [p.position[1] for p in placed]
        span = (max(heights) - min(heights)) if heights else 1.0
        widest = max((p.info.diameter for p in placed), default=1.25)

        root = ConfigNode()
        root.set("ship", self.craft_name)
        root.set("version", ksp_version())
        root.set("description", "Sproektirovano Kerbal Intelligence Agency")
        root.set("type", "VAB")
        root.set("size", f"{widest:.3f},{span:.3f},{widest:.3f}")
        root.set("steamPublishedFileId", 0)
        root.set("persistentId", self._next_uid())
        root.set("rot", "0,0,0,1")
        root.set("missionFlag", "Squad/Flags/default")
        root.set("vesselType", "Ship")

        # Порядок записи: обход дерева в глубину от корня (как это делает KSP)
        order: list[PlacedPart] = []
        self._dfs(placed[0], order)
        for part in order:
            root.children.append(self._render_part(part))
        return root.render() + "\n"

    def _dfs(self, part: PlacedPart, out: list[PlacedPart]) -> None:
        out.append(part)
        for child in part.children:
            self._dfs(child, out)

    def _render_part(self, p: PlacedPart) -> ConfigNode:
        node = ConfigNode(name="PART")
        node.set("part", p.craft_id)
        node.set("partName", "Part")
        node.set("persistentId", p.uid + 1)
        node.set("pos", _vec(p.position))
        node.set("attPos", "0,0,0")
        node.set("attPos0", _vec(p.attach_offset))
        node.set("rot", _quat(p.rotation))
        node.set("attRot", "0,0,0,1")
        node.set("attRot0", _quat(p.rotation))
        node.set("mir", "1,1,1")
        node.set("symMethod", "Radial")
        node.set("autostrutMode", "Off")
        node.set("rigidAttachment", "False")
        node.set("istg", p.stage)
        node.set("resPri", 0)
        node.set("dstg", p.decouple_stage)
        node.set("sidx", p.stage_icon_index)
        node.set("sqor", p.stage_icon_order)
        node.set("sepI", -1)
        node.set("attm", 1 if p.surface_attached else 0)
        node.set("sameVesselCollision", "False")
        node.set("modCost", 0)
        node.set("modMass", 0)
        node.set("modSize", "0,0,0")

        for child in p.children:
            node.set("link", child.craft_id)

        if p.surface_attached and p.parent is not None:
            offset = p.info.surface_node or (0.0, 0.0, 0.0)
            node.set("srfN", f"srfAttach,{p.parent.craft_id},COL,"
                             f"{_pipe(offset)},{_pipe((0.0, -1.0, 0.0))},"
                             f"{_pipe((0.0, 0.0, 0.0))}")

        # attN пишется с обеих сторон соединения (формат KSP 1.12)
        if p.parent is not None and not p.surface_attached:
            node.set("attN", _att_n(p.own_node, p.parent.craft_id, p.info, p.own_node))
        for child in p.children:
            if child.surface_attached:
                continue
            node.set("attN", _att_n(child.parent_node, child.craft_id, p.info,
                                    child.parent_node))

        node.add_node("EVENTS")
        node.add_node("ACTIONS")
        node.add_node("PARTDATA")
        self._render_fairing_shell(node, p)
        return node

    # ------------------------------------------------------------------
    # Оболочка защитного обтекателя
    # ------------------------------------------------------------------
    def _render_fairing_shell(self, node: ConfigNode, p: PlacedPart) -> None:
        """Дописывает створки обтекателя, если это его основание.

        ОБОЛОЧКА — НЕ ДЕТАЛЬ, А НАБОР СЕЧЕНИЙ. В .craft она живёт внутри
        модуля основания списком колец «высота над основанием — радиус».
        Формат снят с чертежа `ComSat Lx.craft` из поставки игры:

            MODULE { name = ModuleProceduralFairing
                XSECTION { h = 0          r = 0.625 }
                XSECTION { h = 1.1307373  r = 0.519284248 }
                XSECTION { h = 1.86900806 r = 0.200000003 } }

        Без этих сечений деталь ставится, но створок нет — обтекателя как
        бы и не существует. Именно это оператор и видел в ангаре.
        """
        if p.info.name != (self._fairing_base.info.name
                           if self._fairing_base else None):
            return
        if p is not self._fairing_base or not self._fairing_sections:
            return
        module = node.add_node("MODULE")
        module.set("name", "ModuleProceduralFairing")
        module.set("isEnabled", "True")
        module.set("interstageCraftID", 0)
        module.set("nArcs", 2)
        module.set("ejectionForce", 100)
        module.set("useClamshell", "False")
        module.set("stagingEnabled", "True")
        module.set("fsm", "st_idle")
        for height, radius in self._fairing_sections:
            section = module.add_node("XSECTION")
            section.set("h", round(height, 6))
            section.set("r", round(radius, 6))

    # ------------------------------------------------------------------
    def save(self, bp: Blueprint, directory: Path | None = None,
             validate: bool = True) -> Path | None:
        directory = directory or self.craft_directory()
        if directory is None:
            log.warning("Каталог VAB не определён — .craft не записан")
            return None
        text = self.render(bp)
        if validate:
            problems = validate_craft(text, self.catalog)
            if problems:
                log.error("Чертёж не прошёл проверку и не записан:")
                for problem in problems:
                    log.error("  ! %s", problem)
                return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.craft_name}.craft"
        path.write_text(text, encoding="utf-8")
        log.info("Чертёж записан: %s (%d деталей)", path, bp.part_count)
        return path

    def craft_directory(self) -> Path | None:
        root = CONFIG.craft.ksp_root
        if not root:
            return None
        return Path(root) / "saves" / CONFIG.craft.save_name / "Ships" / "VAB"


# ==========================================================================
# Проверка чертежа ДО передачи его игре
# ==========================================================================
def validate_craft(text: str, catalog=None) -> list[str]:
    """Ищет в готовом .craft то, на чём споткнётся загрузчик KSP.

    Битый чертёж роняет игру целиком (KSPUtil.GetAttachNodeInfo кидает
    исключение прямо в сцене полёта), поэтому файл проверяется до записи.
    """
    from .config_node import parse

    catalog = catalog or get_catalog()
    problems: list[str] = []
    root = parse(text)
    parts = [n for n in root.children if n.name == "PART"]
    if not parts:
        return ["в чертеже нет ни одной детали"]

    ids = [p.get("part", "") for p in parts]
    if len(set(ids)) != len(ids):
        problems.append("идентификаторы деталей повторяются")

    known = set(ids)
    for part_node, part_id in zip(parts, ids):
        # имя детали должно существовать в игре
        if "_" not in part_id:
            problems.append(f"«{part_id}»: нет разделителя '_' между именем и uid")
            continue
        craft_name, _, uid = part_id.rpartition("_")
        real_name = craft_name.replace(".", "_")
        if not uid.isdigit():
            problems.append(f"«{part_id}»: uid не число")
        if catalog.get(real_name) is None and catalog.get(craft_name) is None:
            problems.append(f"«{part_id}»: детали '{real_name}' нет в каталоге игры")

        for link in part_node.get_all("link"):
            if link not in known:
                problems.append(f"«{part_id}»: link ссылается на неизвестную деталь '{link}'")

        for attn in part_node.get_all("attN"):
            if "," not in attn:
                problems.append(f"«{part_id}»: attN без запятой: '{attn}'")
                continue
            node_id, _, rest = attn.partition(",")
            if not node_id:
                problems.append(f"«{part_id}»: пустое имя узла в attN")
            tokens = rest.split("_")
            # ожидается: имя, uid, позиция, ориентация, исходная позиция, исходная ориентация
            if len(tokens) != 6:
                problems.append(
                    f"«{part_id}»: attN '{attn}' содержит {len(tokens)} полей вместо 6 "
                    f"(формат KSP 1.12 требует четыре вектора)")
                continue
            target = f"{tokens[0]}_{tokens[1]}"
            if target not in known and tokens[0] != "Null":
                problems.append(f"«{part_id}»: attN указывает на неизвестную деталь '{target}'")
            for vector in tokens[2:]:
                if len(vector.split("|")) != 3:
                    problems.append(f"«{part_id}»: вектор '{vector}' не из трёх компонент")

        srf = part_node.get("srfN")
        if srf:
            fields = srf.split(",")
            if len(fields) != 6:
                problems.append(f"«{part_id}»: srfN содержит {len(fields)} полей вместо 6")
            elif fields[1] not in known:
                problems.append(f"«{part_id}»: srfN указывает на неизвестную деталь "
                                f"'{fields[1]}'")

        for key in ("istg", "dstg", "pos", "rot"):
            if part_node.get(key) is None:
                problems.append(f"«{part_id}»: отсутствует обязательное поле {key}")

    # --- связность дерева: от корня можно дойти до каждой детали ---
    children: dict[str, list[str]] = {pid: [] for pid in ids}
    for part_node, part_id in zip(parts, ids):
        for link in part_node.get_all("link"):
            if link in children:
                children[part_id].append(link)
    reachable = set()
    stack = [ids[0]]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(children.get(current, []))
    orphans = [pid for pid in ids if pid not in reachable]
    if orphans:
        problems.append("детали не связаны с корнем (KSP выведет пустую площадку): "
                        + ", ".join(orphans[:5]))

    for part_node, part_id in list(zip(parts, ids))[1:]:
        if not part_node.get_all("attN") and not part_node.get("srfN"):
            problems.append(f"«{part_id}»: деталь ни к чему не прицеплена")

    # --- корень обязан быть командным модулем ---
    root_name = ids[0].rpartition("_")[0].replace(".", "_")
    root_part = catalog.get(root_name)
    if root_part is None:
        problems.append(f"корневая деталь '{root_name}' не найдена в каталоге")
    elif not root_part.is_command:
        problems.append(f"корневая деталь '{root_name}' не командный модуль — "
                        f"аппарат будет неуправляем («Нет управления»)")

    # --- состав: без двигателя и бака ракета не полетит ---
    infos = [catalog.get(pid.rpartition("_")[0].replace(".", "_")) for pid in ids]
    infos = [i for i in infos if i is not None]
    if not any(i.is_engine for i in infos):
        problems.append("в чертеже нет ни одного двигателя")
    if not any(i.is_tank for i in infos):
        problems.append("в чертеже нет топливных баков")
    if not any(i.resources.get("ElectricCharge", 0) > 0 for i in infos):
        problems.append("нет ни одного источника электричества — зонд неуправляем")

    # --- стадии: должен быть хотя бы один запуск двигателя ---
    stages = []
    for part_node in parts:
        try:
            stages.append(int(part_node.get("istg", "-1")))
        except ValueError:
            problems.append("нечисловое значение istg")
    if stages and max(stages) < 0:
        problems.append("ни одна деталь не назначена на стадию — ракета не запустится")

    # --- геометрия: координаты должны быть конечными числами ---
    for part_node, part_id in zip(parts, ids):
        for key in ("pos", "attPos0"):
            raw = part_node.get(key, "")
            for chunk in raw.split(","):
                try:
                    value = float(chunk)
                except ValueError:
                    problems.append(f"«{part_id}»: поле {key} содержит '{chunk}'")
                    break
                if value != value or abs(value) > 1e6:      # NaN / бесконечность
                    problems.append(f"«{part_id}»: недопустимая координата в {key}")
                    break

    # --- скобки: файл должен разбираться обратно без потерь ---
    if text.count("{") != text.count("}"):
        problems.append(f"нарушен баланс фигурных скобок: "
                        f"{text.count('{')} открывающих против {text.count('}')} закрывающих")
    if len(parts) != text.count("PART\n{"):
        problems.append("число блоков PART не совпадает с числом разобранных деталей")

    return problems


# ==========================================================================
def _vec(v) -> str:
    return ",".join(f"{c:.7g}" for c in v)


def _quat(q) -> str:
    return ",".join(f"{c:.7g}" for c in q)


def _pipe(v) -> str:
    return "|".join(f"{c:.9g}" for c in v)


def _att_n(node_name: str, other_id: str, info: PartInfo, own_node_name: str) -> str:
    """Строка attN формата KSP 1.12.

    <узел>,<id детали>_<позиция>_<ориентация>_<исходная позиция>_<исходная ориентация>

    Все четыре вектора относятся к СВОЕЙ детали (той, в блоке которой строка
    записана). Короткая форма из KSP 1.6 в 1.12 роняет загрузчик
    (IndexOutOfRangeException в KSPUtil.GetAttachNodeInfo).
    """
    node = info.node(own_node_name)
    if node is not None:
        pos, orient = node.position, node.orientation
    else:
        half = info.height / 2.0
        pos = (0.0, half if "top" in own_node_name else -half, 0.0)
        orient = (0.0, 1.0 if "top" in own_node_name else -1.0, 0.0)
    body = f"{_pipe(pos)}_{_pipe(orient)}_{_pipe(pos)}_{_pipe(orient)}"
    return f"{node_name},{other_id}_{body}"
