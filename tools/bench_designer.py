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

    total = ok = 0
    failures: dict[str, int] = {}
    for target in targets:
        for kind in KINDS:
            cfg = designer.DesignConfig(kind=kind)
            total += 1
            try:
                design = designer.design(world, target, cfg)
                name = vab.craft_name(design)
                bp = vab.blueprint(design, name)
                asm = CW.CraftAssembler(
                    craft_name=re.sub(r'[<>:"/\\|?*]', "_", name), catalog=parts)
                problems = CW.validate_craft(asm.render(bp), parts)
            except Exception as exc:
                problems = [f"{type(exc).__name__}: {exc}"]
            if problems:
                failures[problems[0][:60]] = failures.get(problems[0][:60], 0) + 1
            else:
                ok += 1

    result = {
        "score": round(100.0 * ok / max(1, total), 1),
        "built": ok, "designs": total,
        "targets": len(targets), "kinds": len(KINDS),
        "school_craft": (school.data().get("dataset", {}) or {}).get("used", 0),
        "failures": dict(sorted(failures.items(), key=lambda kv: -kv[1])[:5]),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"buildable {result['built']}/{result['designs']} "
              f"= {result['score']} %   (school: {result['school_craft']} craft)")
        for reason, count in result["failures"].items():
            print(f"   x{count:<4} {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
