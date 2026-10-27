"""How well the designer builds craft with a given version of the school.

    python tools/bench_designer.py                          installed weights
    python tools/bench_designer.py --school <design_school.json>
    python tools/bench_designer.py --school "" --json       no school at all

For every target of the stock system and every kind of craft it designs a
vehicle, renders the .craft and runs the same validator the game loader
would. Prints how many came out buildable. The game is not needed: the
system is the stock one, the parts come from the cache.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KINDS = ("rocket", "probe", "rover", "station", "base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--school", default=None,
                    help="design_school.json to test; empty string — no school")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from astra import designer, mission, school, vab, world as world_mod
    from astra.craft import catalog
    from kia_core.engineer import craft_writer as CW

    if args.school is not None:
        payload = (json.loads(Path(args.school).read_text(encoding="utf-8"))
                   if args.school else {})
        school.data = lambda: payload          # noqa: E731 — подмена источника

    world = world_mod.detect(connect=False)
    parts = catalog()
    targets = mission.targets(world)

    total = ok = designed = 0
    failures: dict[str, int] = {}
    details: list[dict] = []
    for target in targets:
        for kind in KINDS:
            cfg = designer.DesignConfig(kind=kind)
            total += 1
            try:
                design = designer.design(world, target, cfg)
                if not design.feasible:
                    # Приложение такой проект и не строит: оно пишет
                    # «недостижимо». Считаем это отказом здесь же, чтобы
                    # не мерить сборщик на заведомо пустом чертеже.
                    problems = ["designer gave up: no feasible stage plan"]
                else:
                    designed += 1
                    name = vab.craft_name(design)
                    bp = vab.blueprint(design, name)
                    asm = CW.CraftAssembler(
                        craft_name=re.sub(r'[<>:"/\\|?*]', "_", name),
                        catalog=parts)
                    problems = CW.validate_craft(asm.render(bp), parts)
            except Exception as exc:
                problems = [f"{type(exc).__name__}: {exc}"]
            if problems:
                failures[problems[0][:60]] = failures.get(problems[0][:60], 0) + 1
                stuck = next((s for s in getattr(design, "stages", [])
                              if not s.feasible), None)
                details.append({
                    "target": target.code, "kind": kind,
                    "dv": round(design.budget.total_margin),
                    "stages": len(getattr(design, "stages", [])),
                    "stage": stuck.label if stuck else "",
                    "need_dv": round(stuck.dv_need) if stuck else 0,
                    "need_twr": round(stuck.twr_need, 2) if stuck else 0,
                    "payload_t": round(design.equipment_mass, 2),
                    "why": problems[0][:70],
                })
            else:
                ok += 1

    result = {
        "score": round(100.0 * ok / max(1, total), 1),
        "built": ok, "designs": total, "feasible": designed,
        "targets": len(targets), "kinds": len(KINDS),
        "school_craft": (school.data().get("dataset", {}) or {}).get("used", 0),
        "failures": dict(sorted(failures.items(), key=lambda kv: -kv[1])[:5]),
        "details": details,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"buildable {result['built']}/{result['designs']} "
              f"= {result['score']} %   (school: {result['school_craft']} craft)")
        for reason, count in result["failures"].items():
            print(f"   x{count:<4} {reason}")
        if details:
            print()
            print(f"  {'target':9} {'kind':8} {'dv':>6} {'n':>2} "
                  f"{'stage':9} {'need dv':>8} {'twr':>5} {'payload':>8}")
            for d in details[:20]:
                print(f"  {d['target']:9} {d['kind']:8} {d['dv']:6} "
                      f"{d['stages']:2} {d['stage']:9} {d['need_dv']:8} "
                      f"{d['need_twr']:5} {d['payload_t']:8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
