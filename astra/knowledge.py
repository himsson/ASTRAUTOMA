"""Weights & knowledge: what the AI knows, as separate installable modules.

ASTRAUTOMA ships clean — no weights. Everything the AI knows is added as
modules, one folder each:

    <library>\\<Module>\\<name>\\     one version of a module
        manifest.json                what it is, who made it, what it unlocks
        <files>                      weights (.pt) or knowledge (.json)

Modules:
    FlightProfile   trained flight profile (turn, throttle, burns) —
                    this is what flies the rocket; unlocks mission targets
    Pilot           neural pilot network (ascent control)
    Builder         neural rocket designer
    Math            neural mathematician (maneuver calculations)
    Navigator       interplanetary mission planner
    Knowledge       skills memory and simulator knowledge

The library lives outside the app (by default Desktop\\ASTRAUTOMA Weights)
so the app can be published without weights. Installing copies a module
version into ASTRAUTOMA\\weights\\<Module>. Every .pt file is loaded with
weights_only=True before it is accepted: tensors and numbers only, code
hidden inside a file cannot run.
"""
from __future__ import annotations

import io
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import L

ROOT = Path(__file__).resolve().parent.parent
INSTALLED = ROOT / "weights"
FORMAT = "astra-module-1"

MODULE_ORDER = ("FlightProfile", "Pilot", "Builder", "Math", "Navigator", "Knowledge")

# Which training milestones prove which mission target
TARGET_PROOF = {
    "LEO": "stable_orbit", "HEO": "stable_orbit",
    "LLO": "mun_orbit", "HLO": "mun_orbit",
    "LND": "target_landing",
}


def module_title(module: str) -> str:
    return {
        "FlightProfile": L("Flight profile", "Профиль полёта"),
        "Pilot": L("Neural pilot", "Нейропилот"),
        "Builder": L("Rocket designer", "Конструктор"),
        "Math": L("Mathematician", "Математик"),
        "Navigator": L("Navigator (interplanetary)", "Навигатор (межпланетный)"),
        "Knowledge": L("Knowledge & skills", "Знания и навыки"),
    }.get(module, module)


def module_purpose(module: str) -> str:
    return {
        "FlightProfile": L("flies the rocket; unlocks mission targets",
                           "ведёт ракету; открывает цели полёта"),
        "Pilot": L("ascent control network", "сеть управления выведением"),
        "Builder": L("designs rockets", "проектирует ракеты"),
        "Math": L("maneuver math: Δv, windows, burns", "расчёт манёвров: Δv, окна, импульсы"),
        "Navigator": L("plans trips to planets, comms relays",
                       "планирует полёты к планетам, ретрансляторы"),
        "Knowledge": L("confirmed skills and simulator knowledge",
                       "подтверждённые навыки и знания тренажёров"),
    }.get(module, "")


# ==========================================================================
@dataclass
class ModuleVersion:
    module: str
    name: str
    path: Path
    manifest: dict = field(default_factory=dict)
    error: str = ""

    @property
    def created(self) -> str:
        return self.manifest.get("created", "")

    def summary(self) -> str:
        m = self.manifest
        bits = []
        if m.get("author"):
            bits.append(L("by ", "автор ") + m["author"])
        if m.get("created"):
            bits.append(m["created"])
        if m.get("unlocks"):
            bits.append(L("unlocks ", "открывает ") + ", ".join(m["unlocks"]))
        if m.get("score") is not None:
            bits.append(L(f"score {m['score']:.0f}", f"счёт {m['score']:.0f}"))
        if m.get("accuracy"):
            bits.append(L("accuracy ", "точность ") + m["accuracy"])
        if m.get("skills"):
            bits.append(L("skills: ", "умения: ") + m["skills"])
        return "  ·  ".join(bits)


def library_root(settings: dict) -> Path:
    custom = settings.get("weights_library")
    if custom:
        return Path(custom)
    return Path.home() / "Desktop" / "ASTRAUTOMA Weights"


def _read_version(module: str, folder: Path) -> ModuleVersion:
    mv = ModuleVersion(module, folder.name, folder)
    try:
        mv.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        mv.name = mv.manifest.get("name", folder.name)
        if mv.manifest.get("format") != FORMAT:
            mv.error = L("unknown format", "неизвестный формат")
        elif mv.manifest.get("module") != module:
            mv.error = L("wrong module folder", "не та папка модуля")
    except (OSError, json.JSONDecodeError) as exc:
        mv.error = L(f"no manifest ({exc.__class__.__name__})", f"нет описания ({exc.__class__.__name__})")
    return mv


def scan_library(settings: dict) -> dict[str, list[ModuleVersion]]:
    root = library_root(settings)
    out: dict[str, list[ModuleVersion]] = {m: [] for m in MODULE_ORDER}
    for module in MODULE_ORDER:
        folder = root / module
        if folder.is_dir():
            out[module] = [_read_version(module, f) for f in sorted(folder.iterdir()) if f.is_dir()]
    return out


def installed() -> dict[str, ModuleVersion]:
    out = {}
    for module in MODULE_ORDER:
        folder = INSTALLED / module
        if (folder / "manifest.json").exists():
            out[module] = _read_version(module, folder)
    return out


# ==========================================================================
def _check_file(path: Path) -> None:
    if path.suffix == ".pt":
        import torch
        torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(L(f"file type not allowed: {path.name}", f"недопустимый файл: {path.name}"))


def install(version: ModuleVersion) -> None:
    """Checks every file, then replaces the installed module with this version."""
    if version.error:
        raise ValueError(version.error)
    files = [f for f in version.path.iterdir() if f.is_file() and f.name != "manifest.json"]
    for f in files:
        _check_file(f)
    dest = INSTALLED / version.module
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in files + [version.path / "manifest.json"]:
        shutil.copy2(f, dest / f.name)
    assemble()


def remove(module: str) -> None:
    dest = INSTALLED / module
    if dest.exists():
        shutil.rmtree(dest)
    assemble()


def assemble() -> None:
    """Builds the files the flight code reads from the installed modules.

    kia_core expects one combined network file (kia_model.pth) with the
    pilot, designer and mathematician, plus the navigator next to it.
    They are rebuilt from whatever is installed — and removed when the
    module is gone, so an uninstalled skill really disappears.
    """
    import torch
    mods = installed()
    combined = {}
    stages = {}
    for module, key, stage_key in (("Pilot", "pilot", "pilot"), ("Builder", "builder", "builder"),
                                   ("Math", "mathematician", "math")):
        if module in mods:
            data = torch.load(INSTALLED / module / f"{key}.pt", map_location="cpu", weights_only=True)
            combined[key] = data["state"]
            stages[stage_key] = int(data.get("stage", 1))
    target = ROOT / "kia_model.pth"
    if combined:
        payload = {"version": 2, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                   "stage": stages.get("pilot", 1),
                   "stages": {"builder": stages.get("builder", 1), "math": stages.get("math", 1)}}
        payload.update(combined)
        torch.save(payload, target)
    elif target.exists():
        target.unlink()

    nav_target = ROOT / "kia_model_navigator.pth"
    if "Navigator" in mods:
        shutil.copy2(INSTALLED / "Navigator" / "navigator.pt", nav_target)
    elif nav_target.exists():
        nav_target.unlink()

    skills = ROOT / "kia_math_skills.json"
    if "Math" in mods and (INSTALLED / "Math" / "math_skills.json").exists():
        shutil.copy2(INSTALLED / "Math" / "math_skills.json", skills)


_GENOME_CACHE: dict = {}


def genome() -> dict:
    """Installed flight profile, or {} when none is installed.

    Read once and reused until the file changes: the menu asks for it on
    every redraw, and re-reading JSON each time made the keys lag."""
    path = INSTALLED / "FlightProfile" / "genome.json"
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return {}
    if _GENOME_CACHE.get("stamp") != stamp:
        try:
            _GENOME_CACHE.update(stamp=stamp, data=json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {}
    return _GENOME_CACHE["data"]


def unlocked_targets() -> set[str]:
    """Mission targets the installed weights have proven in training.

    Home and Moon targets — by the flight profile's milestones. A planet —
    only when the Navigator is installed AND the flight profile carries a
    passed interplanetary execution profile for that planet."""
    g = genome()
    milestones = set(g.get("milestones", []))
    out = {code for code, proof in TARGET_PROOF.items() if proof in milestones}
    if "stable_orbit" in milestones and (INSTALLED / "Navigator" / "manifest.json").exists():
        for planet, block in g.get("interplanetary", {}).get("planets", {}).items():
            if block.get("passed"):
                goals = ("LO", "HO") if planet == "Jool" else ("L", "LO", "HO")
                out |= {f"{planet.upper()}-{goal}" for goal in goals}
    # Moons: flown with the same skills as their planet (Minmus — as the Mun),
    # once the flight profile has learned routes between bodies
    from .planets import MOON_OF
    for moon, parent in (MOON_OF.items() if capabilities().get("moons") else ()):
        ok = ("LLO" in out) if parent == "Kerbin" else any(c.startswith(parent.upper() + "-") for c in out)
        if ok:
            out |= {f"{moon.upper()}-L", f"{moon.upper()}-LO"}
    return out


# ==========================================================================
# Building a library from a training folder (Kerbal Intelligence Agency)
# ==========================================================================
def export_from_training(training: Path, library: Path, label: str, author: str = "") -> list[Path]:
    """Splits a training folder's weights into module folders of the library."""
    import torch
    training = Path(training)
    stamp = time.strftime("%Y-%m-%d")
    made = []

    def write(module: str, files: dict[str, bytes | Path], extra: dict) -> None:
        folder = library / module / f"{label} {stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, src in files.items():
            if isinstance(src, Path):
                shutil.copy2(src, folder / name)
            else:
                (folder / name).write_bytes(src)
        manifest = {"format": FORMAT, "module": module, "name": f"{label} {stamp}",
                    "author": author, "created": time.strftime("%Y-%m-%d %H:%M"),
                    "game": "Kerbal Space Program 1.12"}
        manifest.update(extra)
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        made.append(folder)

    best = training / "kia_model_best.pth"
    source = best if best.exists() else training / "kia_model.pth"
    if source.exists():
        payload = torch.load(source, map_location="cpu", weights_only=True)
        stages = payload.get("stages", {})
        for module, key, stage in (("Pilot", "pilot", payload.get("stage", 1)),
                                   ("Builder", "builder", stages.get("builder", 1)),
                                   ("Math", "mathematician", stages.get("math", 1))):
            if key in payload:
                buf = io.BytesIO()
                torch.save({"module": module, "stage": int(stage), "state": payload[key]}, buf)
                files: dict = {f"{key}.pt": buf.getvalue()}
                extra: dict = {"stage": int(stage)}
                if module == "Math" and (training / "kia_math_skills.json").exists():
                    files["math_skills.json"] = training / "kia_math_skills.json"
                write(module, files, extra)

    gen = training / "data" / "best_genome.json"
    if gen.exists():
        data = json.loads(gen.read_text(encoding="utf-8"))
        miles = set(data.get("milestones", []))
        unlocks = [c for c, p in TARGET_PROOF.items() if p in miles]
        # Interplanetary execution profile (cruise simulator) travels inside
        # the flight profile: it is part of how the rocket is flown.
        cruise = training / "data" / "interplanetary_profile.json"
        if cruise.exists():
            data["interplanetary"] = json.loads(cruise.read_text(encoding="utf-8"))
            passed = [n for n, b in data["interplanetary"].get("planets", {}).items() if b.get("passed")]
            unlocks += [f"{n.upper()}-*" for n in passed]
        write("FlightProfile", {"genome.json": json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")},
              {"score": data.get("score"), "unlocks": unlocks,
               "milestones": sorted(miles)})

    nav = training / "kia_model_navigator_best.pth"
    nav = nav if nav.exists() else training / "kia_model_navigator.pth"
    if nav.exists():
        report = training / "data" / "navigator_report.json"
        extra = {}
        if report.exists():
            rep = json.loads(report.read_text(encoding="utf-8"))
            extra = {"stage": rep.get("stage"), "finished": rep.get("finished"),
                     "planets": sorted({k.split("/")[0] for k in rep.get("pairs", {})})}
        write("Navigator", {"navigator.pt": nav}, extra)

    know = {}
    for src, name in (("kia_global_memory.json", "global_memory.json"),
                      ("kia_sim_knowledge.json", "sim_knowledge.json")):
        if (training / src).exists():
            know[name] = training / src
    if know:
        write("Knowledge", know, {})
    return made


# ==========================================================================
# One-click setup: find a downloaded weights archive, unpack it, install all
# ==========================================================================
ALLOWED = {".pt", ".json"}


def find_archives() -> list[Path]:
    """Weights archives lying where people usually save downloads."""
    home = Path.home()
    places = [home / "Downloads", home / "Desktop", ROOT, ROOT.parent]
    seen, out = set(), []
    for d in places:
        try:
            for f in d.glob("*.zip"):
                n = f.name.lower()
                if "astrautoma" in n and "weight" in n and f.resolve() not in seen:
                    seen.add(f.resolve())
                    out.append(f)
        except OSError:
            continue
    return sorted(out, key=lambda f: f.stat().st_mtime, reverse=True)


def import_archive(zip_path: Path, settings: dict) -> int:
    """Unpacks <Module>/<version>/<file> entries into the library.
    Only manifests, .pt and .json files are taken; anything else is skipped."""
    import zipfile
    root = library_root(settings)
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = [p for p in info.filename.replace("\\", "/").split("/") if p]
            # the archive may have one wrapping folder ("ASTRAUTOMA Weights/...")
            while parts and parts[0] not in MODULE_ORDER:
                parts = parts[1:]
            if len(parts) != 3 or ".." in parts or Path(parts[2]).suffix not in ALLOWED:
                continue
            dest = root / parts[0] / parts[1] / parts[2]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
            count += 1
    return count


def newest(versions: list[ModuleVersion]) -> ModuleVersion | None:
    good = [v for v in versions if not v.error]
    return max(good, key=lambda v: (v.created, v.name)) if good else None


def install_all(settings: dict) -> list[str]:
    """Installs the newest good version of every module. Returns what was installed."""
    done = []
    for module, versions in scan_library(settings).items():
        v = newest(versions)
        if v is None:
            continue
        cur = installed().get(module)
        if cur is None or cur.created < v.created or cur.name != v.name:
            install(v)
            done.append(f"{module} · {v.name}")
    return done


def capabilities() -> dict:
    """Skills beyond mission targets, carried by the flight profile:
    moons, return, rendezvous, docking, assists."""
    return dict(genome().get("capabilities", {}))


def assist_atlas() -> dict:
    """Gravity-assist atlas shipped in the Navigator module (or {})."""
    path = INSTALLED / "Navigator" / "assists.json"
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return {}
    if _ATLAS_CACHE.get("stamp") != stamp:
        try:
            _ATLAS_CACHE.update(stamp=stamp, data=json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {}
    return _ATLAS_CACHE["data"]


_ATLAS_CACHE: dict = {}
