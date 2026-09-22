"""Builds the designed rocket as a real .craft file in the VAB.

The designer (designer.py) decides WHAT to fly: stages, engines, tanks,
boosters and the payload kit. This module turns that into a craft the
game opens — every part placed on its real attach nodes by the flight
engine's assembler (kia_core/engineer/craft_writer.py):

    top      parachute or nose cone, command pod
    column   reaction wheel, stack parts of the payload (batteries,
             docking ports, crew cabins, heat shield), adapters
    stages   tanks, engine, decoupler — the top stage first, the
             fairing between the top stage's tank and engine
    sides    panels, antennas, lights, science; landing legs on the top
             stage, rover wheels two rows along the hull; fins on the
             lowest tank; boosters with nose cones; launch clamps

The craft is checked by the same validator the game loader would trip on
before anything is written, and saved to <save>/Ships/VAB, so it shows up
in the VAB's "Load" list.
"""
from __future__ import annotations

import re
from pathlib import Path

from .i18n import L

STACK_KEYS = ("dockingPort", "crewCabin", "batteryBank", "HeatShield", "Large_Crewed_Lab", "sasModule",
              "asasmodule")


def _catalog():
    from .craft import catalog
    return catalog()


def _is_stack(info) -> bool:
    if any(k in info.name for k in STACK_KEYS):
        return True
    return not getattr(info, "can_surface_attach", True)


def blueprint(design, name: str):
    """astra Design → kia_core Blueprint."""
    from kia_core.engineer.craft_writer import Blueprint, BoosterBuild, StageBuild
    cat = _catalog()
    get = cat.get
    req = design.required + [o for o in design.optional if o[1].name.startswith("parachute")][:1]
    pod = chute = fairing = avionics = None
    stack, radial, legs, wheels = [], [], [], []
    for kind, item, _ in req:
        info = get(item.name)
        if info is None:
            continue
        n = item.count
        low = item.name.lower()
        if pod is None and (info.crew_capacity or "probe" in low or "pod" in low) and "cabin" not in low:
            pod = info
            continue
        if low.startswith("fairing"):
            fairing = info
        elif low.startswith("parachute"):
            if not getattr(info, "can_surface_attach", True) or "single" in low:
                chute = chute or info
            else:
                radial += [info] * n
        elif "leg" in low:
            legs += [info] * n
        elif "wheel" in low and "reaction" not in kind.lower() and "sas" not in low:
            wheels += [info] * n
        elif avionics is None and ("sasmodule" in low or "asasmodule" in low):
            avionics = info
            stack += [info] * (n - 1)
        elif _is_stack(info):
            stack += [info] * n
        else:
            radial += [info] * n
    if pod is None:
        raise ValueError(L("no command part in the design", "в проекте нет управляющей детали"))

    stages = []
    ordered = list(design.stages)
    boosters = None
    fins = []
    if ordered and ordered[0].label.lower().startswith(("boost", "ускор")):
        b = ordered.pop(0)
        srb = get(b.engine.name) if b.engine else None
        dec = next((get(x.name) for x in b.extras if "adial" in x.name), None)
        if srb is not None:
            noses = cat.nose_cones(diameter=srb.diameter) or cat.nose_cones()
            boosters = BoosterBuild(booster=srb, decoupler=dec, nose=noses[0] if noses else None,
                                    count=b.engine.count)
    for i, st in enumerate(ordered):
        eng = get(st.engine.name) if st.engine else None
        if eng is None:
            raise ValueError(L(f"{st.label}: no engine", f"{st.label}: нет двигателя"))
        tanks = []
        for t in st.tanks:
            info = get(t.name)
            if info is not None:
                tanks += [info] * t.count
        dec = next((get(x.name) for x in st.extras if x.name.startswith("Decoupler")), None)
        for x in st.extras:
            if x.name in ("basicFin", "R8winglet") and get(x.name):
                fins += [get(x.name)] * x.count
        stages.append(StageBuild(engine=eng, tanks=tanks, decoupler=dec if i > 0 else None, index=i))

    nose = None
    if chute is None:
        cones = cat.nose_cones(diameter=pod.diameter) or []
        nose = cones[0] if cones else None
    adapters = {}
    above = stack[-1] if stack else pod
    for st in sorted(stages, key=lambda s: -s.index):
        if not st.tanks:
            continue
        top_d, bottom_d = above.hull_diameter, st.tanks[0].hull_diameter
        if abs(top_d - bottom_d) > 1e-6:
            found = cat.oriented_adapter(top_d, bottom_d)
            if found is not None:
                adapters[st.index] = found
        above = st.tanks[0]
    clamps = (cat.launch_clamps() or [])[:1] * 4
    return Blueprint(pod=pod, parachute=chute, stages=stages, radial_payload=radial, fins=fins,
                     clamps=clamps, nose=nose, boosters=boosters, adapters=adapters, fairing=fairing,
                     legs=legs, avionics=avionics, payload_stack=stack, rover_wheels=wheels, name=name)


def craft_name(design) -> str:
    kind = getattr(design.config, "kind", "rocket")
    word = {"rocket": "", "rover": " Rover", "base": " Base", "station": " Station", "probe": " Probe"}.get(kind, "")
    back = " R" if getattr(design.target, "return_home", False) else ""
    return f"ASTRA {design.target.code}{word}{back}"


def vab_folder(world) -> Path | None:
    if not world.ksp_root or not world.save_name:
        return None
    return Path(world.ksp_root) / "saves" / world.save_name / "Ships" / "VAB"


def build(world, design, folder: Path | None = None) -> tuple[Path | None, list[str]]:
    """Writes the craft. Returns (path, problems)."""
    from kia_core.engineer import craft_writer as CW
    folder = folder or vab_folder(world)
    if folder is None:
        return None, [L("the save folder is unknown — start the game once", "папка сохранения не найдена — запустите игру")]
    name = craft_name(design)
    try:
        bp = blueprint(design, name)
    except ValueError as exc:
        return None, [str(exc)]
    asm = CW.CraftAssembler(craft_name=re.sub(r'[<>:"/\\|?*]', "_", name), catalog=_catalog())
    text = asm.render(bp)
    problems = CW.validate_craft(text, _catalog())
    if problems:
        return None, problems
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{asm.craft_name}.craft"
    path.write_text(text, encoding="utf-8")
    return path, []
