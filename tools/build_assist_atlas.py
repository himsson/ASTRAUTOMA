"""Builds the gravity-assist atlas that ships in the Navigator weights.

For every planet pair that a mission uses (Kerbin → planet and planet →
Kerbin), departure dates are sampled over many years of game time. At
each date the assist planner searches every flyby planet (thousands of
Lambert arcs) and keeps the best trip only if it beats the best direct
transfer by at least 50 m/s. The app then looks trips up instead of
searching during the flight.

    python tools/build_assist_atlas.py out.json [years]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra import gravity as G              # noqa: E402
from astra.planets import system            # noqa: E402

YEAR = 426 * 21600.0


def main() -> None:
    out = Path(sys.argv[1])
    years = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    s = system()
    planets = [p for p in ("Moho", "Eve", "Duna", "Dres", "Jool", "Eeloo")]
    pairs = [("Kerbin", p) for p in planets] + [(p, "Kerbin") for p in planets]
    atlas = {"format": "astra-assists-1", "years": years, "created": time.strftime("%Y-%m-%d %H:%M"),
             "pairs": {}}
    t0 = time.time()
    for a, b in pairs:
        A, B = s[a], s[b]
        syn = abs(1 / (1 / A.period - 1 / B.period))
        step = syn / 3
        rows = []
        ut = 0.0
        while ut < years * YEAR:
            G._CACHE.clear()
            best = G.best_assist(a, b, ut)
            base = G.direct(a, b, ut)
            if best is not None:
                rows.append({"ut": ut, "flyby": best.flyby, "total": round(best.total, 1),
                             "departure": round(best.departure, 1), "flyby_dv": round(best.flyby_dv, 1),
                             "arrival": round(best.arrival, 1), "t_dep": best.t_dep, "t_flyby": best.t_flyby,
                             "t_arr": best.t_arr, "rp": best.rp, "v_inf_arr": best.v_inf_arr,
                             "direct": round(base.total, 1) if base else None})
            ut += step
        atlas["pairs"][f"{a}>{b}"] = rows
        saved = [r["direct"] - r["total"] for r in rows if r["direct"]]
        print(f"{a:>7} → {b:<7} {len(rows):3d} assists"
              + (f", saves up to {max(saved):.0f} m/s" if saved else "") + f"   [{time.time() - t0:.0f} s]", flush=True)
        out.write_text(json.dumps(atlas, indent=1), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main()
