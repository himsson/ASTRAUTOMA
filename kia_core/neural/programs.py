"""УЧЕБНЫЕ ПРОГРАММЫ КОНСТРУКТОРА И МАТЕМАТИКА.

У пилота своя лестница ступеней (`curriculum.py`) — она про то, как мир
постепенно перестаёт быть удобным. Здесь лестницы двух других сетей, и
устроены они иначе: усложняется не мир, а САМА ЗАДАЧА.

    КОНСТРУКТОР
    1. Сборка      одна цель — орбита Кербина. Научиться ставить
                   ядро, бак, двигатель в осмысленном порядке
    2. Сборка+     цели меняются от эпизода к эпизоду, подключается
                   проектное бюро со своим чек-листом, требуется
                   обвязка: питание, связь, спасение
    3. Сборка++    чертёж обязан РЕАЛЬНО собираться в .craft и
                   проходить валидатор; ΔV считается по ступеням, а не
                   одной кучей; за лишние детали и массу платят

    МАТЕМАТИК
    1. Матан       арифметика ракеты
    2. Матан+      манёвры на орбите
    3. Матан++     межпланетное
    4. Матан+++    сложные манёвры + повторение всего

Почему ступени вообще нужны и здесь. Сеть-конструктор, которой сразу
предъявить полный чек-листа бюро, получает отрицательную награду за любую
сборку и не успевает понять, за что именно, — градиент один и тот же и
для «забыл парашют», и для «поставил двигатель в вакууме внизу».

Экзамен на каждой ступени свой (`pass_mean`, `pass_count`): средняя
награда выше порога И столько-то удачных эпизодов подряд. Это те же
правила, по которым переводится пилот, — см. `pipeline.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import ROOT
from ..logging_setup import get_logger

log = get_logger("neural.programs")

# Снимки весов. Файл целиком (все три сети), как и у пилота: так снимок
# всегда самодостаточен и его можно поднять в любой момент.
BUILDER_STAGE1_FILE = ROOT / "kia_model_builder1.pth"
BUILDER_STAGE2_FILE = ROOT / "kia_model_builder2.pth"
MATH_STAGE1_FILE = ROOT / "kia_model_math1.pth"
MATH_STAGE2_FILE = ROOT / "kia_model_math2.pth"
MATH_STAGE3_FILE = ROOT / "kia_model_math3.pth"


@dataclass(frozen=True)
class TrainingStage:
    """Ступень программы, которая не про полёт."""
    program: str                  # "builder" | "math"
    number: int
    key: str
    title: str
    summary: str
    lessons: tuple[str, ...]
    default_steps: int            # ориентир по эпизодам
    rollout: int
    pass_mean: float              # средняя награда для сдачи
    pass_count: int               # сколько удачных эпизодов нужно
    tolerance: float = 0.0        # только для математика
    backup: Path | None = None    # снимок ПЕРЕД началом ступени

    def describe(self) -> str:
        lines = [f"  {self.number}. {self.title} — {self.summary}"]
        lines += [f"       • {lesson}" for lesson in self.lessons]
        lines.append(f"       ориентир {self.default_steps} эпизодов, "
                     f"экзамен: средняя ≥ {self.pass_mean:+.2f} и "
                     f"{self.pass_count} удачных")
        if self.tolerance:
            lines.append(f"       допуск ответа {self.tolerance * 100:.0f} %")
        return "\n".join(lines)


# ==========================================================================
BUILDER_STAGES: tuple[TrainingStage, ...] = (
    TrainingStage(
        program="builder", number=1, key="build", title="Сборка",
        summary="одна цель: собрать то, что вообще летает",
        lessons=(
            "первым делом ядро управления — без него аппарат мёртв",
            "двигателю нужен бак, баку нужен двигатель",
            "тяга должна превышать вес, иначе ракета не тронется",
            "лишние детали — это масса, за которую платят топливом",
        ),
        default_steps=6_000, rollout=512,
        pass_mean=1.60, pass_count=400, backup=None,
    ),
    TrainingStage(
        program="builder", number=2, key="build_plus", title="Сборка+",
        summary="цель меняется, и за чертёж отвечают перед бюро",
        lessons=(
            "задача каждый раз новая: орбита, перелёт, посадка",
            "проектное бюро проверяет устойчивость и запас топлива",
            "беспилотнику нужны питание и связь, иначе управления нет",
            "возврат с атмосферного тела требует парашюта",
        ),
        default_steps=12_000, rollout=512,
        pass_mean=1.90, pass_count=600, backup=BUILDER_STAGE1_FILE,
    ),
    TrainingStage(
        program="builder", number=3, key="build_pro", title="Сборка++",
        summary="чертёж обязан собираться в игре и считаться по ступеням",
        lessons=(
            "сборка переводится в .craft и проходит валидатор — "
            "несобираемое не засчитывается вовсе",
            "ΔV считается ПО СТУПЕНЯМ: верхняя ступень не тащит "
            "топливо нижней, и одной кучей это не сходится",
            "за каждую лишнюю деталь и каждую лишнюю тонну штраф",
            "стартовый TWR проверяется по нижней ступени, а не по всей "
            "массе сразу",
        ),
        default_steps=20_000, rollout=512,
        pass_mean=2.20, pass_count=800, backup=BUILDER_STAGE2_FILE,
    ),
)

MATH_STAGES: tuple[TrainingStage, ...] = (
    TrainingStage(
        program="math", number=1, key="math", title="Матан",
        summary="арифметика ракеты",
        lessons=(
            "круговая скорость и период обращения",
            "ускорение силы тяжести на высоте",
            "Циолковский: сколько ΔV даёт заправка",
            "тяговооружённость и время ожога",
        ),
        default_steps=3_000_000, rollout=1024,
        pass_mean=0.90, pass_count=2_000, tolerance=0.05, backup=None,
    ),
    TrainingStage(
        program="math", number=2, key="math_plus", title="Матан+",
        summary="манёвры на орбите",
        lessons=(
            "vis-viva: скорость в любой точке эллипса",
            "подъём апоапсиса и скругление — два импульса Гомана",
            "время перелёта",
            "куда вынесет аппарат при текущей скорости",
            "допуск ужесточён с 5 % до 3 %",
        ),
        default_steps=5_000_000, rollout=1024,
        pass_mean=0.85, pass_count=3_000, tolerance=0.03,
        backup=MATH_STAGE1_FILE,
    ),
    TrainingStage(
        program="math", number=3, key="math_pro", title="Матан++",
        summary="межпланетное",
        lessons=(
            "фазовый угол: когда открывается окно",
            "синодический период: как часто оно открывается",
            "смена плоскости орбиты",
            "уход из сферы влияния и захват у цели",
            "допуск 2 % — половина от первой ступени",
        ),
        default_steps=8_000_000, rollout=1024,
        pass_mean=0.85, pass_count=4_000, tolerance=0.02,
        backup=MATH_STAGE2_FILE,
    ),
    TrainingStage(
        program="math", number=4, key="math_max", title="Матан+++",
        summary="сложные манёвры и повторение всего",
        lessons=(
            "двухэллиптический переход: три импульса",
            "ΔV двухступенчатой ракеты с полезной нагрузкой",
            "уход с опорной орбиты с избытком v∞ — эффект Оберта",
            "скругление и смена плоскости одним импульсом",
            "посадка в последний момент: высота и время торможения",
            "радиус сферы действия",
            "половина задач — повторение трёх прошлых ступеней",
        ),
        default_steps=12_000_000, rollout=2048,
        pass_mean=0.90, pass_count=6_000, tolerance=0.02,
        backup=MATH_STAGE3_FILE,
    ),
)

PROGRAMS: dict[str, tuple[TrainingStage, ...]] = {
    "builder": BUILDER_STAGES,
    "math": MATH_STAGES,
}


# ==========================================================================
def stages_of(program: str) -> tuple[TrainingStage, ...]:
    if program not in PROGRAMS:
        raise KeyError(f"нет программы «{program}»")
    return PROGRAMS[program]


def resolve(program: str, reference: str | int) -> TrainingStage:
    """Ступень по номеру или короткому имени."""
    stages = stages_of(program)
    text = str(reference).strip().lower()
    if text.isdigit():
        number = int(text)
        for stage in stages:
            if stage.number == number:
                return stage
        raise KeyError(f"нет ступени №{number} в программе «{program}»")
    for stage in stages:
        if stage.key == text:
            return stage
    raise KeyError(f"нет ступени «{reference}» в программе «{program}»")


def prepare(program: str, stage: TrainingStage, brain,
            overwrite_backup: bool = False) -> list[str]:
    """Готовит веса к обучению на ступени: снимок прошлой и пометка новой.

    Правило то же, что у пилота: снимок делается ОДИН раз и сам собой не
    перезаписывается. Повторный вход на ту же ступень продолжает обучение,
    а не откатывает его.
    """
    import shutil

    messages: list[str] = []
    reached = int(brain.stages.get(program, 1))
    continuing = reached >= stage.number

    if continuing:
        messages.append(f"{program}: уже на ступени {reached} — продолжаем")
    elif reached < stage.number - 1:
        messages.append(f"{program}: ступень {stage.number - 1} пропущена "
                        f"(пройдено {reached}) — перехожу всё равно")

    if stage.backup is not None and not continuing:
        source = brain.path
        if not source.exists():
            messages.append(f"нет файла весов {source.name} — снимок не сделан")
        elif stage.backup.exists() and not overwrite_backup:
            messages.append(f"снимок {stage.backup.name} уже есть — не трогаю")
        else:
            shutil.copy2(source, stage.backup)
            messages.append(f"снимок {source.name} -> {stage.backup.name}")
            log.info("Снимок программы %s: %s -> %s", program, source,
                     stage.backup)

    brain.stages[program] = stage.number
    return messages


def rollback(program: str, stage: TrainingStage, brain) -> bool:
    """Поднимает снимок указанной ступени обратно в работу."""
    stages = stages_of(program)
    snapshot = None
    for candidate in stages:
        if candidate.number == stage.number + 1:
            snapshot = candidate.backup
    if snapshot is None or not snapshot.exists():
        log.warning("Снимка ступени %d программы %s нет", stage.number, program)
        return False
    if not brain.load(snapshot):
        return False
    brain.stages[program] = stage.number
    brain.save()
    log.info("Программа %s откачена на ступень %d", program, stage.number)
    return True


def describe_program(program: str) -> str:
    titles = {"builder": "УЧЕБНАЯ ПРОГРАММА КОНСТРУКТОРА",
              "math": "УЧЕБНАЯ ПРОГРАММА МАТЕМАТИКА"}
    lines = [titles.get(program, program.upper()), ""]
    for stage in stages_of(program):
        lines.append(stage.describe())
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = ["TrainingStage", "BUILDER_STAGES", "MATH_STAGES", "PROGRAMS",
           "stages_of", "resolve", "prepare", "rollback", "describe_program",
           "BUILDER_STAGE1_FILE", "BUILDER_STAGE2_FILE",
           "MATH_STAGE1_FILE", "MATH_STAGE2_FILE", "MATH_STAGE3_FILE"]
