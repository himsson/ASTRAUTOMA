"""Что на ракете и что когда от неё отделяется.

Порядок расстыковки не нужно угадывать: он записан в самом аппарате.
У каждой детали есть `decouple_stage` — номер ступени, на которой она
уходит, и `activate_stage` — когда включается двигатель. Детали, которые
не уходят никогда, и есть полезный груз: то, ради чего летели.

Отсюда и разбор: разложить аппарат по группам отделения, назвать каждую
по тому, из чего она состоит, и показать оператору до старта. Тогда
«отстыковать вот это вот здесь» перестаёт быть вопросом — это уже
записано в чертеже, остаётся прочитать вслух.

Модуль ничего не решает и ни на что не влияет: он только читает.
"""
from __future__ import annotations

from .i18n import L


def _is_tank(part) -> bool:
    return (part.resources_max.get("LiquidFuel", 0) > 0
            or part.resources_max.get("SolidFuel", 0) > 0)


def name_group(parts: list) -> str:
    """Как назвать эту связку деталей по тому, из чего она состоит."""
    engines = [p for p in parts if p.thrust_vac > 0]
    tanks = [p for p in parts if _is_tank(p)]
    legs = [p for p in parts if p.leg]
    chutes = [p for p in parts if p.parachute]
    crew = sum(p.crew_capacity for p in parts)
    relay = [p for p in parts if p.relay]
    antenna = [p for p in parts if p.antenna_power > 0]

    if crew:
        return L(f"crew capsule ({crew} seats)", f"кабина экипажа ({crew} мест)")
    if legs and engines:
        return L("lander", "посадочный модуль")
    if relay:
        return L("relay satellite", "спутник-ретранслятор")
    if engines and tanks:
        return L("stage", "ступень")
    if antenna or any(p.solar_rate > 0 for p in parts):
        return L("probe", "зонд")
    if chutes:
        return L("descent module", "спускаемый аппарат")
    if tanks:
        return L("tanks", "баки")
    return L("equipment", "оборудование")


def contents(parts: list) -> str:
    """Короткий перечень: чего и сколько в связке."""
    counts = [
        (L("engine", "двигатель"), sum(1 for p in parts if p.thrust_vac > 0)),
        (L("tank", "бак"), sum(1 for p in parts if _is_tank(p))),
        (L("leg", "опора"), sum(1 for p in parts if p.leg)),
        (L("parachute", "парашют"), sum(1 for p in parts if p.parachute)),
        (L("antenna", "антенна"), sum(1 for p in parts if p.antenna_power > 0)),
        (L("panel", "панель"), sum(1 for p in parts if p.solar_rate > 0)),
    ]
    return ", ".join(f"{word} x{n}" for word, n in counts if n)


def groups(vessel) -> list[tuple[int, list]]:
    """Детали по ступеням отделения, снизу вверх. -1 — то, что остаётся."""
    by_stage: dict[int, list] = {}
    for part in vessel.parts:
        by_stage.setdefault(int(part.decouple_stage), []).append(part)
    ordered = sorted((s for s in by_stage if s >= 0), reverse=True)
    out = [(s, by_stage[s]) for s in ordered]
    if -1 in by_stage:
        out.append((-1, by_stage[-1]))
    return out


def describe(vessel) -> list[str]:
    """Строки для предстартового отчёта: что уходит и что остаётся."""
    rows = groups(vessel)
    if len(rows) < 2:
        return []
    lines = []
    for stage, parts in rows:
        if stage < 0:
            lines.append(L(f"stays in flight — {name_group(parts)}: {contents(parts)}",
                           f"остаётся в полёте — {name_group(parts)}: {contents(parts)}"))
        else:
            lines.append(L(f"stage {stage} separates — {name_group(parts)}: {contents(parts)}",
                           f"ступень {stage} отделяется — {name_group(parts)}: {contents(parts)}"))
    return lines
