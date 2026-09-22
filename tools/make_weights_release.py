"""Packs a weights release: one version of every module into one zip.

    python tools/make_weights_release.py <source version folder name> <new version> <out.zip> [assists.json] [Module=file ...]

Takes <library>\\<Module>\\<source>\\ for every module, renames it to
"ASTRAUTOMA <new version>", adds the skills this version unlocks to the
flight profile and the gravity-assist atlas to the navigator, and writes
<Module>/ASTRAUTOMA <version>/<files> into the zip — the layout the app
imports with one key.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

LIB = Path.home() / "Desktop" / "ASTRAUTOMA Weights"

CAPABILITIES = {"moons": True, "return": True, "rendezvous": True, "docking": True, "assists": True}
MOON_UNLOCKS = ["MINMUS-*", "IKE-*", "GILLY-*", "LAYTHE-*", "VALL-*", "TYLO-*", "BOP-*", "POL-*"]
ACCURACY = {"Math": "99.3 / 97.7 / 95.1 / 94.9 % by level",
            "Navigator": "97.6 % of mission choices (ceiling 98.2 %)"}
SKILLS = {"FlightProfile": "moons, return home, rendezvous, docking, gravity assists",
          "Navigator": "gravity-assist atlas: 12 planet pairs, 20 years",
          "Builder": "design school: stages, TWR, Δv split, engines and kits learned from real craft"}


def main() -> None:
    src, version, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    rest = sys.argv[4:]
    atlas = Path(rest[0]) if rest and "=" not in rest[0] else None
    extra: dict[str, list[Path]] = {}
    for arg in rest:
        if "=" in arg:
            mod, f = arg.split("=", 1)
            extra.setdefault(mod, []).append(Path(f))
    name = f"ASTRAUTOMA {version}"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for module in ("FlightProfile", "Pilot", "Builder", "Math", "Navigator", "Knowledge"):
            folder = LIB / module / src
            if not folder.is_dir():
                raise SystemExit(f"missing {folder}")
            base = f"{module}/{name}"
            replaced = {f.name for f in extra.get(module, [])}
            for f in folder.iterdir():
                if f.name == "manifest.json" or f.name in replaced:
                    continue
                if module == "FlightProfile" and f.name == "genome.json":
                    g = json.loads(f.read_text(encoding="utf-8"))
                    g["capabilities"] = CAPABILITIES
                    z.writestr(f"{base}/genome.json", json.dumps(g, ensure_ascii=False, indent=2))
                    continue
                z.write(f, f"{base}/{f.name}")
            if module == "Navigator" and atlas is not None:
                z.write(atlas, f"{base}/assists.json")
            for f in extra.get(module, []):
                z.write(f, f"{base}/{f.name}")
            m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            m.update(name=name, created=time.strftime("%Y-%m-%d %H:%M"), author="himsson")
            if module in ACCURACY:
                m["accuracy"] = ACCURACY[module]
            if module in SKILLS:
                m["skills"] = SKILLS[module]
            if module == "FlightProfile":
                m["unlocks"] = list(dict.fromkeys(list(m.get("unlocks", [])) + MOON_UNLOCKS))
            z.writestr(f"{base}/manifest.json", json.dumps(m, ensure_ascii=False, indent=2))
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
