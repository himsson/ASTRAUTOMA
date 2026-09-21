"""Отчёт конструктора: kia_design_report.md.

ИИ обязан объяснить свои решения человеку: какие формулы применил, какие
числа получил, почему выбрал именно эти детали и где остался запас.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..config import ROOT
from ..logging_setup import get_logger
from .autodesign import VehicleDesign

log = get_logger("engineer.report")

REPORT_MD = ROOT / "kia_design_report.md"
REPORT_JSON = ROOT / "kia_design_report.json"


def write_report(design: VehicleDesign, md_path: Path | None = None,
                 json_path: Path | None = None) -> Path:
    md_path = Path(md_path or REPORT_MD)
    json_path = Path(json_path or REPORT_JSON)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(design), encoding="utf-8")
    json_path.write_text(json.dumps(design.summary(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    log.info("Отчёт конструктора записан: %s", md_path)
    return md_path


def render_markdown(design: VehicleDesign) -> str:
    spec, budget, policy = design.spec, design.budget, design.policy
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = []
    add = L.append

    add("# Отчёт конструктора KIA")
    add("")
    add(f"**Дата расчёта:** {now}  ")
    add(f"**Задача:** {spec.title} — {spec.describe()}  ")
    if spec.raw_command:
        add(f"**Команда оператора:** «{spec.raw_command}»  ")
    add(f"**Источник деталей:** каталог установленной игры "
        f"({len(_catalog_size(design))} деталей прочитано из cfg)  ")
    add(f"**Вердикт:** {'✅ проект жизнеспособен' if design.viable else '❌ проект отклонён'}")
    add("")

    # ------------------------------------------------------------------
    add("## 1. Расчёт требуемой характеристической скорости")
    add("")
    add("Бюджет считается из орбитальной механики под конкретную задачу, "
        "а не берётся из готовой таблицы.")
    add("")
    add("| Участок | Формула | Числа | ΔV, м/с |")
    add("|---|---|---|---|")
    for leg in budget.legs:
        add(f"| {leg.name} | `{leg.formula}` | {leg.numbers} | **{leg.dv:.0f}** |")
    add(f"| | | Сумма | **{budget.raw_total:.0f}** |")
    add(f"| | Запас ×{budget.margin:.2f} | Итого требуется | **{budget.total:.0f}** |")
    add("")

    # ------------------------------------------------------------------
    add("## 2. Полезная нагрузка")
    add("")
    add("| Деталь | Роль | Масса, т | Почему выбрана |")
    add("|---|---|---|---|")
    for choice in design.payload:
        add(f"| {choice.part.title} `{choice.part.name}` | {choice.role} | "
            f"{choice.part.dry_mass:.3f} | {choice.reason} |")
    add(f"| | **Итого** | **{design.payload_mass:.3f}** | |")
    add("")

    # ------------------------------------------------------------------
    add("## 3. Разбиение на ступени и подбор деталей")
    add("")
    add("Решается обратная задача Циолковского: для каждой пары "
        "«двигатель × бак» ищется минимальное число баков, дающее нужный ΔV, "
        "затем проверяется TWR, из прошедших выбирается вариант наименьшей массы.")
    add("")
    add("```")
    add("ΔV = Isp · g₀ · ln(m_полная / m_сухая)        (формула Циолковского)")
    add("TWR = F / (m · g)                             (тяга к весу)")
    add("t_ожога = (m_полная - m_сухая) / (F / (Isp·g₀))")
    add("```")
    add("")

    for s in sorted(design.stages, key=lambda x: -x.requirement.index):
        req = s.requirement
        add(f"### Ступень {req.index} — {req.role}")
        add("")
        add(f"**Требование:** ΔV ≥ {req.dv_required:.0f} м/с, TWR ≥ {req.min_twr:.2f} "
            f"(g = {req.gravity:.2f} м/с², среднее давление {req.pressure:.2f} атм)")
        add("")
        for line in s.rationale:
            add(f"- {line}")
        add("")
        add("| Параметр | Значение |")
        add("|---|---|")
        add(f"| Двигатель | {s.engine.title} (`{s.engine.name}`) |")
        add(f"| Тяга на участке | {s.thrust():.1f} кН |")
        add(f"| Удельный импульс | {s.isp():.0f} с |")
        add(f"| Баки | {s.tank_count} × {s.tank.title} (`{s.tank.name}`) |")
        add(f"| Разделитель | {s.decoupler.title if s.decoupler else '— (нижняя ступень)'} |")
        add(f"| Масса нагрузки сверху | {s.payload_mass:.3f} т |")
        add(f"| Сухая масса ступени в сборе | {s.dry_mass:.3f} т |")
        add(f"| Полная масса ступени в сборе | {s.wet_mass:.3f} т |")
        add(f"| Топливо | {s.fuel_mass:.3f} т |")
        add(f"| Расчётный ΔV | **{s.delta_v():.0f} м/с** (нужно {req.dv_required:.0f}) |")
        add(f"| Расчётный TWR | **{s.twr():.2f}** (нужно {req.min_twr:.2f}) |")
        add(f"| Время ожога | {s.burn_time():.0f} с |")
        add(f"| Просмотрено вариантов | {s.candidates_examined} |")
        add("")

    # ------------------------------------------------------------------
    pack = design.boosters
    if pack is not None:
        add("### Боковые ускорители")
        add("")
        for line in pack.rationale:
            add(f"- {line}")
        add("")
        add("| Параметр | Значение |")
        add("|---|---|")
        add(f"| Ускоритель | {pack.part.title} (`{pack.part.name}`) |")
        add(f"| Количество | **{pack.count}** (симметрично) |")
        add(f"| Разделитель | {pack.decoupler.title if pack.decoupler else '—'} |")
        add(f"| Обтекатель ускорителя | {pack.nose.title if pack.nose else '—'} |")
        add(f"| Тяга одного | {pack.thrust_each:.0f} кН |")
        add(f"| Суммарная тяга | **{pack.total_thrust:.0f} кН** |")
        add(f"| Масса заправленного | {pack.wet_each:.2f} т |")
        add(f"| Суммарная масса | {pack.total_wet:.2f} т |")
        add(f"| Время горения | {pack.burn_time:.0f} с |")
        add(f"| TWR центрального блока | {pack.core_twr:.2f} |")
        add(f"| TWR связки на старте | **{pack.twr:.2f}** |")
        add(f"| Прибавка ΔV (в бюджет не засчитана) | +{pack.delta_v_bonus:.0f} м/с |")
        add("")

    add("## 4. Итог по аппарату")
    add("")
    add("| Параметр | Значение |")
    add("|---|---|")
    add(f"| Стартовая масса | **{design.total_mass:.2f} т** |")
    add(f"| Масса полезной нагрузки | {design.payload_mass:.3f} т |")
    add(f"| Суммарный ΔV | **{design.total_delta_v:.0f} м/с** |")
    add(f"| Требуется по задаче | {design.budget.total:.0f} м/с |")
    add(f"| Запас | {design.total_delta_v - design.budget.total:+.0f} м/с |")
    add(f"| Стартовый TWR | **{design.launch_twr:.2f}** |")
    add(f"| Число ступеней | {len(design.stages)} |")
    add(f"| Число деталей | {design.part_count} |")
    add("")

    if design.problems:
        add("### Проблемы")
        add("")
        for p in design.problems:
            add(f"- ❌ {p}")
        add("")
    if design.warnings:
        add("### Замечания")
        add("")
        for w in design.warnings:
            add(f"- ⚠️ {w}")
        add("")

    # ------------------------------------------------------------------
    add("## 5. Политика проектирования (настраивается обучением)")
    add("")
    add("| Параметр | Значение | Смысл |")
    add("|---|---|---|")
    meanings = {
        "ascent_split": "доля ΔV подъёма, отданная стартовой ступени",
        "liftoff_twr": "целевой стартовый TWR",
        "upper_twr": "минимальный TWR разгонных ступеней",
        "transfer_twr": "минимальный TWR перелётной ступени",
        "dv_margin": "запас ΔV сверх расчёта",
        "mass_penalty": "вес массы при выборе варианта",
        "part_penalty": "штраф за число деталей",
        "science_level": "сколько научных приборов брать",
        "solar_level": "брать ли солнечные панели",
        "fin_level": "аэродинамическая стабилизация оперением",
        "booster_level": "насколько старт переложен на боковые ускорители",
    }
    for key, value in policy.to_dict().items():
        add(f"| `{key}` | {value:.3f} | {meanings.get(key, '')} |")
    add("")
    add("Эти параметры эволюционируют между запусками: успешные комбинации "
        "закрепляются, неудачные вытесняются.")
    add("")
    add("## 6. Сборка")
    add("")
    add(f"Чертёж `{design.blueprint.name}.craft` собирается по реальным "
        f"стыковочным узлам деталей: баки ставятся на срез двигателя, "
        f"между ступенями всегда стоит разделитель, мелкие приборы крепятся "
        f"радиально на верхний бак.")
    add("")
    add("Порядок стадий (первой срабатывает наибольшая):")
    add("")
    n = len(design.stages)
    for s in sorted(design.stages, key=lambda x: -x.requirement.index):
        i = s.requirement.index
        add(f"- стадия {2 * (n - i) - 1}: запуск двигателя ступени {i} "
            f"({s.engine.title})")
        if i > 0:
            add(f"- стадия {2 * (n - i) - 2}: отделение ступени {i - 1}")
    add("- стадия 0: раскрытие парашюта")
    add("")
    return "\n".join(L) + "\n"


def _catalog_size(design: VehicleDesign) -> list:
    try:
        from .parts_catalog import get_catalog
        return list(get_catalog().parts)
    except Exception:
        return []
