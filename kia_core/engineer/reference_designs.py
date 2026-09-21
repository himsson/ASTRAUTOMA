"""ОБУЧЕНИЕ НА ОБРАЗЦАХ: чему KIA учится у готовых чертежей.

Своя конструкторская школа у KIA сложилась узкой. Она считает Циолковского
и подбирает баки с двигателями, но весь её опыт происходит из одного
шаблона «ядро — баки — двигатель — оперение». Когда такой аппарат
оказывается неуправляемым, крутить она может только те ручки, которые у
неё есть. Заметить ОТСУТСТВУЮЩУЮ деталь ей неоткуда: в её мире рулевых
двигателей и обтекателей просто не существует.

Готовый чертёж, который заведомо летает, это ограничение снимает. Отсюда
берутся не «правила от разработчика», а НОРМЫ, снятые с образцов:

    сколько маховиков несут аппараты такой массы;
    с какой массы начинают ставить RCS;
    с какой массы обязателен обтекатель;
    какое суммарное качание сопел считается нормальным.

Замер, ради которого модуль написан. Аппарат KIA массой 9 т имел один
маховик и ОДНО сопло с качанием 4°, а на части чертежей — вообще 0°,
потому что угол качания не участвовал в подборе двигателя. Для сравнения,
образцы сопоставимой массы:

    Спутник связи SWM-94, 6.8 т: 2 маховика, три сопла по 22.5°
    Зоркий, 4.0 т:              3 маховика

Ракета KIA не отрабатывала команду тангажа (просят 90°, стоит на 70°) —
не потому что пилот плох, а потому что рулить было нечем.

Нормы пересчитываются командой `python tools/analyze_craft.py --learn`
и лежат в data/reference_norms.json. Если образцов нет, конструктор
работает как раньше: модуль ничего не навязывает.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import DATA_DIR
from ..logging_setup import get_logger

log = get_logger("engineer.reference")

NORMS_FILE = DATA_DIR / "reference_norms.json"

# Граница между «мелочью» и «серьёзным аппаратом», тонн. Взята не с
# потолка: на образцах ровно по ней проходит раздел — легче неё ни один
# не несёт ни RCS, ни обтекателя, тяжелее несут почти все.
HEAVY_MASS = 50.0

# Доля образцов, при которой признак считается НОРМОЙ, а не особенностью
# конкретного аппарата. Две трети — то есть «так делают почти все».
NORM_SHARE = 0.66


@dataclass
class ReferenceNorms:
    """Что образцы считают само собой разумеющимся."""
    sample_count: int = 0
    sample_names: list = field(default_factory=list)

    # Маховики: сколько несут аппараты лёгкого и тяжёлого класса
    wheels_light: int = 1
    wheels_heavy: int = 2
    # С какой массы образцы начинают ставить RCS и обтекатель
    rcs_from_mass: float = 0.0          # 0 — не ставят никогда
    fairing_from_mass: float = 0.0
    boosters_from_mass: float = 0.0
    # Суммарное качание сопел, которое образцы считают достаточным
    gimbal_light: float = 0.0
    gimbal_heavy: float = 0.0

    def wheels_for(self, mass: float) -> int:
        return self.wheels_heavy if mass >= HEAVY_MASS else self.wheels_light

    def gimbal_for(self, mass: float) -> float:
        return self.gimbal_heavy if mass >= HEAVY_MASS else self.gimbal_light

    def wants_rcs(self, mass: float) -> bool:
        return bool(self.rcs_from_mass) and mass >= self.rcs_from_mass

    def wants_fairing(self, mass: float) -> bool:
        return bool(self.fairing_from_mass) and mass >= self.fairing_from_mass

    def wants_boosters(self, mass: float) -> bool:
        return bool(self.boosters_from_mass) and mass >= self.boosters_from_mass

    @property
    def known(self) -> bool:
        return self.sample_count > 0

    def describe(self) -> str:
        if not self.known:
            return "Образцов нет — конструктор работает на собственных правилах."
        lines = [f"Нормы, снятые с {self.sample_count} образцов "
                 f"({', '.join(self.sample_names[:4])}"
                 f"{' и др.' if len(self.sample_names) > 4 else ''}):"]
        lines.append(f"  маховиков: {self.wheels_light} на лёгком аппарате, "
                     f"{self.wheels_heavy} на тяжелее {HEAVY_MASS:.0f} т")
        lines.append(f"  суммарное качание сопел: {self.gimbal_light:.0f}° "
                     f"лёгкий, {self.gimbal_heavy:.0f}° тяжёлый")
        for label, value in (("RCS", self.rcs_from_mass),
                             ("обтекатель", self.fairing_from_mass),
                             ("ускорители", self.boosters_from_mass)):
            lines.append(f"  {label}: " + (f"начиная с {value:.0f} т"
                                           if value else "образцы не ставят"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceNorms":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ==========================================================================
def learn(folder: Path, catalog=None) -> ReferenceNorms:
    """Снимает нормы с чертежей в папке. Свои чертежи KIA пропускаются."""
    import sys
    tools = Path(__file__).resolve().parent.parent.parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from analyze_craft import parse_craft
    from .parts_catalog import get_catalog

    catalog = catalog or get_catalog()
    norms = ReferenceNorms()
    reports = []
    from ..config import CONFIG

    own = str(getattr(CONFIG.craft, "craft_name", "KIA_Autogen") or "KIA")
    for path in sorted(Path(folder).glob("*.craft")):
        try:
            report = parse_craft(path, catalog)
        except Exception as exc:
            log.debug("Чертёж %s не разобран: %s", path.name, exc)
            continue
        # На себе не учатся. Проверять надо И имя файла, И имя корабля
        # внутри: свой чертёж, сохранённый оператором под другим именем,
        # иначе попадёт в образцы, и система начнёт закреплять собственные
        # ошибки как норму.
        if path.stem.startswith("KIA") or report.name.startswith(("KIA", own)):
            continue
        if len(report.parts) < 5:
            continue                       # пустышка или автосохранение
        reports.append(report)

    if not reports:
        log.info("Образцов для обучения не найдено в %s", folder)
        return norms

    norms.sample_count = len(reports)
    norms.sample_names = [r.name for r in reports]
    heavy = [r for r in reports if r.wet_mass >= HEAVY_MASS]
    light = [r for r in reports if r.wet_mass < HEAVY_MASS]

    def median(values, default=0.0):
        vals = sorted(values)
        return vals[len(vals) // 2] if vals else default

    norms.wheels_light = int(median([r.reaction_wheels for r in light], 1)) or 1
    norms.wheels_heavy = int(median([r.reaction_wheels for r in heavy], 2)) or 2
    norms.gimbal_light = float(median(
        [sum(e.gimbal for e in r.engines) for r in light if r.engines], 0.0))
    norms.gimbal_heavy = float(median(
        [sum(e.gimbal for e in r.engines) for r in heavy if r.engines], 0.0))

    # «Начиная с какой массы» — самый лёгкий образец, который признак несёт,
    # но только если признак встречается достаточно часто, чтобы считаться
    # нормой, а не причудой одного аппарата.
    def threshold(has) -> float:
        carriers = [r.wet_mass for r in reports if has(r)]
        if not carriers:
            return 0.0
        share_heavy = (sum(1 for r in heavy if has(r)) / len(heavy)
                       if heavy else 0.0)
        if share_heavy < NORM_SHARE:
            return 0.0
        return min(carriers)

    norms.rcs_from_mass = threshold(lambda r: r.rcs_blocks > 0)
    norms.fairing_from_mass = threshold(lambda r: r.fairings > 0)
    norms.boosters_from_mass = threshold(lambda r: bool(r.boosters))

    log.info("Нормы сняты с %d образцов: маховиков %d/%d, качание %.0f/%.0f°, "
             "RCS с %.0f т, обтекатель с %.0f т", norms.sample_count,
             norms.wheels_light, norms.wheels_heavy,
             norms.gimbal_light, norms.gimbal_heavy,
             norms.rcs_from_mass, norms.fairing_from_mass)
    return norms


def save(norms: ReferenceNorms, path: Path | None = None) -> Path:
    target = Path(path or NORMS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(norms.to_dict(), ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


_CACHE: ReferenceNorms | None = None


def load(path: Path | None = None, refresh: bool = False) -> ReferenceNorms:
    """Нормы из файла. Без файла возвращает пустые — они ничего не требуют."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    target = Path(path or NORMS_FILE)
    if not target.exists():
        _CACHE = ReferenceNorms()
        return _CACHE
    try:
        _CACHE = ReferenceNorms.from_dict(
            json.loads(target.read_text(encoding="utf-8")))
    except Exception as exc:
        log.warning("Нормы не прочитаны (%s) — работаю без образцов", exc)
        _CACHE = ReferenceNorms()
    return _CACHE


__all__ = ["ReferenceNorms", "learn", "save", "load", "NORMS_FILE",
           "HEAVY_MASS"]
