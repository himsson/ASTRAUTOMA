"""Мир игры: где стоит KSP, какое сохранение загружено и какие у него настройки.

Приложение подстраивается под мир, а не под константы:
  * параметры тел (μ, радиус, атмосфера, период вращения, орбиты лун)
    берутся из живой игры через kRPC — так учитываются Kopernicus,
    Sigma Dimensions, RSS и любые другие масштабы;
  * без связи с игрой — стандартная система KSP 1.12;
  * настройки сложности (CommNet, дальность связи, нагрев, режим игры,
    уровень станции слежения) читаются из persistent.sfs сохранения.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import L

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "config.json"

DEFAULT_KSP_ROOTS = (
    r"D:\SteamLibrary\steamapps\common\Kerbal Space Program",
    r"C:\Program Files (x86)\Steam\steamapps\common\Kerbal Space Program",
    r"C:\Program Files\Steam\steamapps\common\Kerbal Space Program",
    r"E:\SteamLibrary\steamapps\common\Kerbal Space Program",
)

# Мощность наземной сети (DSN) по уровню станции слежения, как в KSP 1.12
DSN_POWER = {1: 2.0e9, 2: 5.0e10, 3: 2.5e11}


# ==========================================================================
@dataclass
class BodyInfo:
    name: str
    mu: float
    radius: float
    rotation_period: float
    atmosphere_depth: float = 0.0
    soi: float = math.inf
    parent: str | None = None
    sma: float = 0.0                 # большая полуось орбиты вокруг родителя
    inclination: float = 0.0         # рад
    lan: float = 0.0                 # долгота восходящего узла, рад
    period: float = 0.0
    max_terrain: float = 0.0         # высота самой высокой точки, м
    satellites: list[str] = field(default_factory=list)

    @property
    def g0(self) -> float:
        return self.mu / self.radius ** 2

    @property
    def rotation_speed(self) -> float:
        """Линейная скорость вращения поверхности на экваторе, м/с."""
        if not self.rotation_period:
            return 0.0
        return 2 * math.pi * self.radius / self.rotation_period

    def v_circ(self, altitude: float) -> float:
        return math.sqrt(self.mu / (self.radius + altitude))

    def period_at(self, altitude: float) -> float:
        a = self.radius + altitude
        return 2 * math.pi * math.sqrt(a ** 3 / self.mu)

    def synchronous_altitude(self) -> float:
        if not self.rotation_period:
            return math.inf
        a = (self.mu * (self.rotation_period / (2 * math.pi)) ** 2) ** (1 / 3)
        return a - self.radius


# Стандартная система KSP 1.12 (только то, что нужно для целей приложения)
STOCK_BODIES = {
    "Kerbin": BodyInfo("Kerbin", 3.5316e12, 600_000.0, 21_549.425, 70_000.0,
                       84_159_286.0, "Sun", 13_599_840_256.0, 0.0, 0.0,
                       9_203_545.0, 6_764.1, ["Mun", "Minmus"]),
    "Mun": BodyInfo("Mun", 6.5138398e10, 200_000.0, 138_984.38, 0.0,
                    2_429_559.1, "Kerbin", 12_000_000.0, 0.0, 0.0,
                    138_984.38, 7_061.0),
    "Minmus": BodyInfo("Minmus", 1.7658e9, 60_000.0, 40_400.0, 0.0,
                       2_247_428.4, "Kerbin", 47_000_000.0, math.radians(6.0),
                       math.radians(78.0), 1_077_310.5, 5_725.0),
}
STOCK_SITE = {"name": "KSC", "latitude": -0.0972, "longitude": -74.5577}

# Высшие точки рельефа стоковых тел (kRPC их не отдаёт)
STOCK_TERRAIN = {"Mun": 7_061.0, "Minmus": 5_725.0, "Moon": 10_786.0}


# ==========================================================================
@dataclass
class WorldSettings:
    """Настройки сохранения, влияющие на полёт."""
    mode: str = "SANDBOX"
    version: str = ""
    modded: bool = False
    commnet: bool = True
    require_signal: bool = False
    plasma_blackout: bool = False
    range_modifier: float = 1.0
    dsn_modifier: float = 1.0
    ground_stations: bool = True
    reentry_heat: float = 1.0
    tracking_level: int = 3
    ut: float = 0.0

    @property
    def dsn_power(self) -> float:
        return DSN_POWER.get(self.tracking_level, 2.0e9) * self.dsn_modifier


@dataclass
class World:
    ksp_root: Path | None = None
    save_name: str = ""
    save_path: Path | None = None
    save_time: float = 0.0
    settings: WorldSettings = field(default_factory=WorldSettings)
    bodies: dict[str, BodyInfo] = field(default_factory=dict)
    home: str = "Kerbin"
    moon: str = "Mun"
    site: dict = field(default_factory=lambda: dict(STOCK_SITE))
    live: bool = False                 # параметры тел сняты с игры
    connection: object = None          # KRPCConnection или None
    scene: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def home_body(self) -> BodyInfo:
        return self.bodies[self.home]

    @property
    def moon_body(self) -> BodyInfo:
        return self.bodies[self.moon]

    @property
    def scaled(self) -> bool:
        return abs(self.home_body.radius - 600_000.0) > 1_000.0


# ==========================================================================
def load_settings() -> dict:
    defaults = {"ksp_root": "", "save_name": "auto", "krpc_address": "127.0.0.1",
                "rpc_port": 50000, "stream_port": 50001, "logo_style": "braille"}
    local = ROOT / "config.local.json"
    if local.exists():
        try:
            data = json.loads(local.read_text(encoding="utf-8"))
            defaults["ksp_root"] = data.get("craft", {}).get("ksp_root", "")
            conn = data.get("connection", {})
            defaults["krpc_address"] = conn.get("address", defaults["krpc_address"])
            defaults["rpc_port"] = conn.get("rpc_port", defaults["rpc_port"])
            defaults["stream_port"] = conn.get("stream_port", defaults["stream_port"])
        except (OSError, json.JSONDecodeError):
            pass
    if SETTINGS_FILE.exists():
        try:
            defaults.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    else:
        SETTINGS_FILE.write_text(json.dumps(defaults, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return defaults


def find_ksp_root(configured: str) -> Path | None:
    for candidate in (configured, *DEFAULT_KSP_ROOTS):
        if candidate and (Path(candidate) / "saves").is_dir():
            return Path(candidate)
    return None


def find_save(ksp_root: Path, wanted: str, vessel_name: str | None = None
              ) -> tuple[str, Path] | None:
    """Загруженное сохранение.

    KSP пишет persistent.sfs при каждой смене сцены, поэтому у сохранения,
    в котором сейчас играют, файл самый свежий. Если игра подключена и
    аппарат в полёте — предпочитается сохранение, где этот аппарат есть.
    """
    saves = ksp_root / "saves"
    if wanted and wanted != "auto":
        path = saves / wanted / "persistent.sfs"
        return (wanted, path) if path.exists() else None
    found = [(p.stat().st_mtime, p.parent.name, p)
             for p in saves.glob("*/persistent.sfs")
             if p.parent.name not in ("training", "scenarios")]
    if not found:
        return None
    found.sort(reverse=True)
    if vessel_name:
        needle = f"name = {vessel_name}"
        for _, name, path in found[:6]:
            try:
                if needle in path.read_text(encoding="utf-8", errors="ignore"):
                    return name, path
            except OSError:
                continue
    return found[0][1], found[0][2]


def read_settings(sfs: Path) -> WorldSettings:
    s = WorldSettings()
    try:
        with sfs.open(encoding="utf-8", errors="ignore") as fh:
            head = fh.read(60_000)
        text = sfs.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return s

    def val(key: str, block: str = head) -> str | None:
        m = re.search(rf"^\s*{key}\s*=\s*(.*?)\s*$", block, re.M)
        return m.group(1) if m else None

    def flag(key: str, default: bool) -> bool:
        v = val(key)
        return default if v is None else v.strip().lower() == "true"

    def num(key: str, default: float) -> float:
        try:
            return float(val(key))
        except (TypeError, ValueError):
            return default

    s.mode = (val("Mode") or "SANDBOX").upper()
    s.version = (val("versionFull") or "").split(" ")[0]
    s.modded = flag("modded", False)
    s.commnet = flag("EnableCommNet", True)
    s.require_signal = flag("requireSignalForControl", False)
    s.plasma_blackout = flag("plasmaBlackout", False)
    s.range_modifier = num("rangeModifier", 1.0)
    s.dsn_modifier = num("DSNModifier", 1.0)
    s.ground_stations = flag("enableGroundStations", True)
    s.reentry_heat = num("ReentryHeatScale", 1.0)
    ut = re.search(r"^\s*UT\s*=\s*([\d.]+)", text, re.M)
    s.ut = float(ut.group(1)) if ut else 0.0
    if s.mode == "SANDBOX":
        s.tracking_level = 3
    else:
        m = re.search(r"TrackingStation\s*\{\s*lvl\s*=\s*([\d.]+)", text)
        lvl = float(m.group(1)) if m else 0.0
        s.tracking_level = 1 + int(round(lvl * 2))
    return s


# ==========================================================================
def bodies_from_game(conn) -> tuple[dict[str, BodyInfo], str, dict]:
    """Снимает параметры тел с живой игры."""
    sc = conn.space_center
    out: dict[str, BodyInfo] = {}
    for name, b in sc.bodies.items():
        try:
            orbit = b.orbit
        except Exception:
            orbit = None
        info = BodyInfo(
            name=name, mu=b.gravitational_parameter, radius=b.equatorial_radius,
            rotation_period=b.rotational_period,
            atmosphere_depth=b.atmosphere_depth if b.has_atmosphere else 0.0,
            soi=b.sphere_of_influence,
            parent=(orbit.body.name if orbit is not None else None),
            sma=(orbit.semi_major_axis if orbit is not None else 0.0),
            inclination=(orbit.inclination if orbit is not None else 0.0),
            lan=(orbit.longitude_of_ascending_node if orbit is not None else 0.0),
            period=(orbit.period if orbit is not None else 0.0),
            max_terrain=STOCK_TERRAIN.get(name, 0.0),
            satellites=[s.name for s in b.satellites])
        out[name] = info
    # Дом — тело, у которого стоит стартовый стол
    home = "Kerbin"
    site = dict(STOCK_SITE)
    try:
        for s in sc.launch_sites:
            if s.name.lower().startswith("launchpad") or s.name.upper() == "LAUNCHPAD":
                home = s.body.name
                break
    except Exception:
        pass
    try:
        v = sc.active_vessel
        sit = str(v.situation).split(".")[-1]
        if sit in ("pre_launch", "landed", "splashed") and v.orbit.body.name in out \
                and out[v.orbit.body.name].atmosphere_depth > 0:
            home = v.orbit.body.name
            flight = v.flight(v.orbit.body.reference_frame)
            site = {"name": "Launch site", "latitude": flight.latitude,
                    "longitude": flight.longitude}
    except Exception:
        pass
    if home not in out:
        home = next(iter(out))
    return out, home, site


def pick_moon(bodies: dict[str, BodyInfo], home: str) -> str:
    """«Луна» — самый массивный спутник родной планеты."""
    sats = [bodies[s] for s in bodies[home].satellites if s in bodies]
    if not sats:
        sats = [b for b in bodies.values() if b.parent == home]
    if not sats:
        return home
    return max(sats, key=lambda b: b.mu).name


def terrain_estimate(body: BodyInfo) -> float:
    """Высшая точка рельефа: табличная или оценка по радиусу."""
    if body.max_terrain:
        return body.max_terrain
    return body.radius * 0.035


# ==========================================================================
def _port_open(address: str, port: int) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(0.6)
    try:
        return s.connect_ex((address, port)) == 0
    finally:
        s.close()


def detect(settings: dict | None = None, connect: bool = True) -> World:
    settings = settings or load_settings()
    world = World()
    world.bodies = {k: BodyInfo(**{**v.__dict__, "satellites": list(v.satellites)})
                    for k, v in STOCK_BODIES.items()}

    vessel_name = None
    if connect and not _port_open(settings["krpc_address"], int(settings["rpc_port"])):
        world.notes.append(L("kRPC is not answering — stock KSP system "
                             "(start KSP and press Start Server in the kRPC window)",
                             "kRPC не отвечает — стандартная система KSP "
                             "(запустите KSP и Start Server в окне kRPC)"))
        connect = False
    if connect:
        try:
            from kia_core.config import CONFIG
            CONFIG.connection.name = "ASTRAUTOMA"
            CONFIG.connection.retry_attempts = 1
            CONFIG.connection.address = settings["krpc_address"]
            CONFIG.connection.rpc_port = int(settings["rpc_port"])
            CONFIG.connection.stream_port = int(settings["stream_port"])
            from kia_core.environment.connection import KRPCConnection
            conn = KRPCConnection().connect()
            world.connection = conn
            world.scene = conn.game_scene()
            world.bodies, world.home, world.site = bodies_from_game(conn)
            world.live = True
            if conn.in_flight():
                vessel_name = conn.active_vessel.name
        except Exception as exc:
            world.notes.append(L(f"kRPC unavailable ({type(exc).__name__}) — stock KSP system",
                                 f"kRPC недоступен ({type(exc).__name__}) — стандартная система KSP"))

    world.moon = pick_moon(world.bodies, world.home)
    world.ksp_root = find_ksp_root(settings.get("ksp_root", ""))
    if world.ksp_root is None:
        world.notes.append(L("KSP folder not found — set ksp_root in config.json", "Папка KSP не найдена — укажите ksp_root в config.json"))
        return world
    save = find_save(world.ksp_root, settings.get("save_name", "auto"), vessel_name)
    if save is None:
        world.notes.append(L("No saves found", "Сохранения не найдены"))
        return world
    world.save_name, world.save_path = save
    world.save_time = world.save_path.stat().st_mtime
    world.settings = read_settings(world.save_path)
    return world


def reconnect(world: World, settings: dict | None = None) -> World:
    if world.connection is not None:
        try:
            world.connection.close()
        except Exception:
            pass
    return detect(settings)
