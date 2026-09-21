"""Телеметрия: потоки данных kRPC в реальном времени.

Все «горячие» величины читаются через streams (сервер сам шлёт обновления),
редкие — прямыми RPC-вызовами. Снимок состояния (`snapshot()`) отдаёт
плоский dict, пригодный для функции наград и логов.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("env.telemetry")

RESOURCE_NAMES = ("LiquidFuel", "Oxidizer", "SolidFuel", "ElectricCharge",
                  "MonoPropellant", "Ablator")


@dataclass
class TelemetrySnapshot:
    ut: float = 0.0
    mission_time: float = 0.0
    body: str = ""
    situation: str = ""

    altitude: float = 0.0            # над уровнем моря
    surface_altitude: float = 0.0    # над рельефом
    apoapsis: float = 0.0
    periapsis: float = 0.0
    eccentricity: float = 0.0
    inclination: float = 0.0
    semi_major_axis: float = 0.0
    orbital_period: float = 0.0
    time_to_apoapsis: float = 0.0
    time_to_periapsis: float = 0.0

    speed: float = 0.0               # орбитальная
    surface_speed: float = 0.0
    vertical_speed: float = 0.0
    horizontal_speed: float = 0.0
    prograde: tuple = (0.0, 0.0, 0.0)
    retrograde: tuple = (0.0, 0.0, 0.0)

    pitch: float = 0.0
    heading: float = 0.0
    roll: float = 0.0
    angle_of_attack: float = 0.0
    sideslip_angle: float = 0.0
    aero_angles_known: bool = False   # игра дала углы сама, а не мы их считали
    dynamic_pressure: float = 0.0
    static_pressure: float = 0.0
    atmosphere_density: float = 0.0
    g_force: float = 0.0

    mass: float = 0.0
    dry_mass: float = 0.0
    thrust: float = 0.0
    available_thrust: float = 0.0
    max_thrust: float = 0.0
    specific_impulse: float = 0.0
    throttle: float = 0.0
    twr: float = 0.0
    stage: int = 0
    part_count: int = 0

    resources: dict = field(default_factory=dict)
    stage_resources: dict = field(default_factory=dict)
    crew_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Telemetry:
    """Держатель потоков kRPC для активного судна."""

    def __init__(self, connection):
        self.connection = connection
        self.sc = connection.space_center
        self.vessel = self.sc.active_vessel
        self.body = self.vessel.orbit.body
        self.body_ref = self.body.reference_frame
        self.surface_ref = self.vessel.surface_reference_frame
        self.orbital_ref = self.vessel.orbital_reference_frame
        self.flight = self.vessel.flight(self.body_ref)
        self.srf_flight = self.vessel.flight(self.vessel.surface_reference_frame)
        # Невращающаяся система планеты: только в ней скорость орбитальная,
        # то есть такая же, как считает тренажёр (см. _open_streams)
        self.orbital_flight = self.vessel.flight(
            self.body.non_rotating_reference_frame)
        self._streams: dict[str, object] = {}
        self._open_streams()
        # Последнее УДАЧНО прочитанное число деталей. Нужно, чтобы сбой
        # чтения не выглядел как разрушение аппарата (см. snapshot).
        self._last_part_count = self._safe(
            lambda: len(self.vessel.parts.all), 0)
        log.info("Телеметрия активирована для судна '%s' (%d деталей)",
                 self.vessel.name, self._last_part_count)

    # ------------------------------------------------------------------
    def _add(self, key, target, attr) -> None:
        try:
            self._streams[key] = self.connection.conn.add_stream(getattr, target, attr)
        except Exception as exc:
            log.debug("Поток '%s' недоступен: %s", key, exc)

    def _open_streams(self) -> None:
        conn, v, orbit, flight = self.connection.conn, self.vessel, self.vessel.orbit, self.flight
        self._streams["ut"] = conn.add_stream(getattr, self.sc, "ut")
        self._add("mission_time", v, "met")
        self._add("mass", v, "mass")
        self._add("dry_mass", v, "dry_mass")
        self._add("thrust", v, "thrust")
        self._add("available_thrust", v, "available_thrust")
        self._add("max_thrust", v, "max_thrust")
        self._add("specific_impulse", v, "specific_impulse")

        self._add("altitude", flight, "mean_altitude")
        self._add("surface_altitude", flight, "surface_altitude")
        # СКОРОСТИ — ТОЛЬКО ОРБИТАЛЬНЫЕ, из НЕВРАЩАЮЩЕЙСЯ системы.
        #
        # `body.reference_frame` вращается вместе с планетой, и скорость
        # в нём — это скорость относительно поверхности: на стартовом
        # столе ровно ноль. Тренажёр же считает орбитальную скорость, и у
        # него на столе 175 м/с — вращение Кербина. Сеть, обученная на
        # такой, в игре видела ноль и снова оказывалась в состоянии,
        # которого в обучении не было ни разу.
        self._add("speed", self.orbital_flight, "speed")
        self._add("vertical_speed", self.orbital_flight, "vertical_speed")
        self._add("horizontal_speed", self.orbital_flight, "horizontal_speed")
        # УГЛЫ — ТОЛЬКО ИЗ ПОВЕРХНОСТНОЙ СИСТЕМЫ ОТСЧЁТА.
        #
        # `flight` привязан к системе планеты, и «тангаж» там означает
        # совсем не то, что в тренажёре. Замер на живой игре: ракета
        # стоит на столе строго вертикально (физическая ось корпуса
        # вверх +1.000), поверхностный тангаж 89.91° — а этот поток
        # отдавал 15.35°.
        #
        # Сеть, обученная на «90 градусов = вертикально», видела 15 и
        # решала, что аппарат почти лежит. Она задирала нос изо всех сил,
        # аппарат перекашивало и он врезался в стартовую площадку. Первый
        # живой полёт нейросети закончился так за две секунды.
        #
        # Скорости, наоборот, остаются в системе планеты: в тренажёре
        # `speed` — это орбитальная скорость с вращением Кербина (175 м/с
        # уже на столе), и поверхностная сюда не годится.
        self._add("pitch", self.srf_flight, "pitch")
        self._add("heading", self.srf_flight, "heading")
        self._add("roll", self.srf_flight, "roll")
        # Угол атаки и скольжение игра считает сама и честнее, чем их
        # можно восстановить из тангажа: в них учтён и курс, и вращение
        # атмосферы. В тренажёре скольжение наказывается по обеим осям
        # (`hypot(тангаж, рыскание)`), и в игре должно быть так же —
        # иначе сеть придёт с политикой под другую функцию награды
        # (раздел 5.4a MIGRATION). Если сборка kRPC этих свойств не даёт,
        # поток просто не откроется, и `flight_env` посчитает по-старому.
        self._add("angle_of_attack", flight, "angle_of_attack")
        self._add("sideslip_angle", flight, "sideslip_angle")
        self._add("dynamic_pressure", flight, "dynamic_pressure")
        self._add("static_pressure", flight, "static_pressure")
        self._add("atmosphere_density", flight, "atmosphere_density")
        self._add("g_force", flight, "g_force")

        self._add("apoapsis", orbit, "apoapsis_altitude")
        self._add("periapsis", orbit, "periapsis_altitude")
        self._add("eccentricity", orbit, "eccentricity")
        self._add("inclination", orbit, "inclination")
        self._add("semi_major_axis", orbit, "semi_major_axis")
        self._add("orbital_period", orbit, "period")
        self._add("time_to_apoapsis", orbit, "time_to_apoapsis")
        self._add("time_to_periapsis", orbit, "time_to_periapsis")

        # То, что раньше спрашивалось отдельным RPC на каждом такте цикла
        # управления (20 раз в секунду), — теперь потоками: каждый лишний
        # запрос съедает кадр игры, и на большой ракете FPS падал до 5.
        self._add("situation_raw", v, "situation")
        try:
            self._add("prograde", v.flight(self.orbital_ref), "prograde")
        except Exception:
            pass
        self._add("crew_count", v, "crew_count")
        self._add("current_stage", v.control, "current_stage")
        self._add("throttle", v.control, "throttle")

        # РЕСУРСЫ — НЕ ПОТОКАМИ. Поток kRPC пересчитывается игрой каждый
        # кадр, а подсчёт ресурса обходит все детали: замер на «Наполлоне»
        # (156 деталей) — 10–34 мс на поток, 12 потоков ≈ 250 мс на кадр,
        # FPS 160 -> 5. Остатки читаются обычным запросом в _resources().

        log.debug("Открыто потоков: %d", len(self._streams))

    def get(self, key: str, default=0.0):
        stream = self._streams.get(key)
        if stream is None:
            return default
        try:
            return stream()
        except Exception:
            return default

    def close(self) -> None:
        for name, stream in list(self._streams.items()):
            try:
                stream.remove()
            except Exception:
                pass
        self._streams.clear()
        log.debug("Потоки телеметрии закрыты")

    def rebind(self) -> None:
        """Переоткрывает потоки — нужно после смены судна или SOI."""
        self.close()
        self.__dict__.pop("_ttl_cache", None)
        self.__dict__.pop("_res_cache", None)
        self.vessel = self.sc.active_vessel
        self.body = self.vessel.orbit.body
        self.body_ref = self.body.reference_frame
        self.surface_ref = self.vessel.surface_reference_frame
        self.orbital_ref = self.vessel.orbital_reference_frame
        self.flight = self.vessel.flight(self.body_ref)
        # поверхностный и орбитальный объекты тоже обязаны смотреть на
        # НОВОЕ судно: из первого читаются углы, из второго — скорости
        self.srf_flight = self.vessel.flight(self.surface_ref)
        self.orbital_flight = self.vessel.flight(
            self.body.non_rotating_reference_frame)
        self._open_streams()
        log.info("Телеметрия перепривязана: тело=%s", self.body.name)

    # ------------------------------------------------------------------
    # Производные величины
    # ------------------------------------------------------------------
    def current_twr(self) -> float:
        mass = self.get("mass", 0.0)
        if mass <= 0:
            return 0.0
        g = self._cached("surface_gravity", 5.0,
                         lambda: self.vessel.orbit.body.surface_gravity, 9.81)
        return self.get("available_thrust", 0.0) / (mass * g)

    def prograde_vector(self) -> tuple:
        if "prograde" in self._streams:
            try:
                return tuple(self._streams["prograde"]())
            except Exception:
                pass
        try:
            return tuple(self.vessel.flight(self.orbital_ref).prograde)
        except Exception:
            return (0.0, 0.0, 0.0)

    def retrograde_vector(self) -> tuple:
        p = self.prograde_vector()
        return (-p[0], -p[1], -p[2])

    def velocity(self, reference_frame=None):
        ref = reference_frame or self.body_ref
        return tuple(self.vessel.velocity(ref))

    def position(self, reference_frame=None):
        ref = reference_frame or self.body_ref
        return tuple(self.vessel.position(ref))

    def resources_by_stage(self, stage: int, cumulative: bool = False) -> dict:
        out = {}
        try:
            res = self.vessel.resources_in_decouple_stage(stage=stage, cumulative=cumulative)
            for name in res.names:
                out[name] = {"amount": res.amount(name), "max": res.max(name)}
        except Exception as exc:
            log.debug("Не удалось прочитать ресурсы ступени %d: %s", stage, exc)
        return out

    def all_stage_resources(self) -> dict:
        current = self.current_stage()
        return {str(s): self.resources_by_stage(s)
                for s in range(max(0, current - 4), current + 1)}

    def current_stage(self) -> int:
        if "current_stage" in self._streams:
            try:
                return int(self._streams["current_stage"]())
            except Exception:
                pass
        try:
            return self.vessel.control.current_stage
        except Exception:
            return 0

    # Какие ресурсы реально нужны пилотам и учёту вех
    TRACKED_RESOURCES = ("LiquidFuel", "Oxidizer", "SolidFuel", "ElectricCharge")

    def _resources(self) -> dict:
        """Остатки ресурсов обычным запросом, не чаще раза в 2 с.

        Подсчёт ресурса игрой обходит все детали (10–34 мс на большой
        ракете), поэтому ни потоков, ни опроса на каждом такте: топливо
        за две секунды не кончится незаметно.
        """
        now = time.monotonic()
        cache = self.__dict__.setdefault("_res_cache", {"t": -1e9, "data": {},
                                                        "tmax": -1e9, "max": {}})
        if now - cache["t"] < 2.0:
            return cache["data"]
        cache["t"] = now
        data = {}
        try:
            res = self.vessel.resources
            # Ёмкость баков (20 мс на запрос) меняется только при отделении
            # ступени — перечитываем её раз в 10 с.
            if now - cache["tmax"] > 10.0:
                cache["tmax"] = now
                names = set(res.names)
                cache["max"] = {r: res.max(r) for r in self.TRACKED_RESOURCES if r in names}
            for r, mx in cache["max"].items():
                data[r] = {"amount": res.amount(r), "max": mx}
        except Exception:
            return cache["data"]
        cache["data"] = data
        return data

    def _cached(self, key, ttl, fn, default=None):
        """Редко меняющееся значение: спрашиваем игру не чаще раза в ttl с."""
        cache = self.__dict__.setdefault("_ttl_cache", {})
        now = time.monotonic()
        hit = cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
        try:
            value = fn()
        except Exception:
            return hit[1] if hit is not None else default
        cache[key] = (now, value)
        return value

    def situation(self) -> str:
        if "situation_raw" in self._streams:
            try:
                return str(self._streams["situation_raw"]()).split(".")[-1]
            except Exception:
                pass
        try:
            return str(self.vessel.situation).split(".")[-1]
        except Exception:
            return "unknown"

    def phase_angle(self, target_name: str) -> float:
        """Фазовый угол (градусы) между судном и целевым телом вокруг общего родителя."""
        try:
            target = self.sc.bodies[target_name]
            parent_ref = target.orbit.body.reference_frame
            v_pos = self.vessel.position(parent_ref)
            t_pos = target.position(parent_ref)
            a = math.atan2(v_pos[2], v_pos[0])
            b = math.atan2(t_pos[2], t_pos[0])
            angle = math.degrees(b - a) % 360.0
            return angle
        except Exception as exc:
            log.debug("Не удалось вычислить фазовый угол до %s: %s", target_name, exc)
            return float("nan")

    def distance_to_body(self, body_name: str) -> float:
        try:
            body = self.sc.bodies[body_name]
            pos = self.vessel.position(body.reference_frame)
            return math.sqrt(sum(c * c for c in pos))
        except Exception:
            return float("inf")

    def is_in_soi(self, body_name: str) -> bool:
        try:
            return self.vessel.orbit.body.name == body_name
        except Exception:
            return False

    # ------------------------------------------------------------------
    def snapshot(self, detailed: bool = False) -> TelemetrySnapshot:
        """Срез состояния.

        detailed=True дополнительно опрашивает ресурсы по ступеням — это
        десятки RPC-вызовов, поэтому в циклах управления не используется.
        """
        resources = self._resources()

        body_name = self._cached("body_name", 1.0, lambda: self.vessel.orbit.body.name, "")

        return TelemetrySnapshot(
            ut=self.get("ut"),
            mission_time=self.get("mission_time"),
            body=body_name,
            situation=self.situation(),
            altitude=self.get("altitude"),
            surface_altitude=self.get("surface_altitude"),
            apoapsis=self.get("apoapsis"),
            periapsis=self.get("periapsis"),
            eccentricity=self.get("eccentricity"),
            inclination=self.get("inclination"),
            semi_major_axis=self.get("semi_major_axis"),
            orbital_period=self.get("orbital_period"),
            time_to_apoapsis=self.get("time_to_apoapsis"),
            time_to_periapsis=self.get("time_to_periapsis"),
            speed=self.get("speed"),
            surface_speed=self.get("speed"),
            vertical_speed=self.get("vertical_speed"),
            horizontal_speed=self.get("horizontal_speed"),
            prograde=self.prograde_vector(),
            retrograde=self.retrograde_vector(),
            pitch=self.get("pitch"),
            heading=self.get("heading"),
            roll=self.get("roll"),
            angle_of_attack=self.get("angle_of_attack"),
            sideslip_angle=self.get("sideslip_angle"),
            aero_angles_known=("angle_of_attack" in self._streams
                               and "sideslip_angle" in self._streams),
            dynamic_pressure=self.get("dynamic_pressure"),
            static_pressure=self.get("static_pressure"),
            atmosphere_density=self.get("atmosphere_density"),
            g_force=self.get("g_force"),
            # kRPC отдаёт массу в кг, тягу в Н — приводим к тоннам и кН
            mass=self.get("mass") / 1000.0,
            dry_mass=self.get("dry_mass") / 1000.0,
            thrust=self.get("thrust") / 1000.0,
            available_thrust=self.get("available_thrust") / 1000.0,
            max_thrust=self.get("max_thrust") / 1000.0,
            specific_impulse=self.get("specific_impulse"),
            throttle=self.get("throttle", 0.0),
            twr=self.current_twr(),
            stage=self.current_stage(),
            # Число деталей: при неудачном чтении отдаём ПРОШЛОЕ значение,
            # а не ноль. Ноль означает «аппарата больше нет», и `flight_env`
            # честно объявлял по нему разрушение — а ссылка на судно
            # становится недействительной после каждого отделения (в KSP
            # отстрелённые части становятся отдельными объектами). Первый
            # же живой полёт нейросети закончился «аппарат разрушен» через
            # две секунды после отстрела мачт, при том что ракета стояла
            # целая. Не смог прочитать — так и скажи, а не хорони аппарат.
            part_count=self._remember_parts(),
            resources=resources,
            stage_resources=self.all_stage_resources() if detailed else {},
            crew_count=int(self.get("crew_count", 0)),
        )

    @staticmethod
    def _safe(fn, default=0.0):
        try:
            return fn()
        except Exception:
            return default

    def _remember_parts(self) -> int:
        """Число деталей, переживающее временный сбой чтения.

        Ссылка на судно становится недействительной после каждого
        отделения: в KSP отстрелённые части превращаются в отдельные
        объекты, и поток по старому судну падает. Отдавать в этот момент
        ноль нельзя — по нему `flight_env` объявляет разрушение аппарата.
        Первый живой полёт нейросети так и закончился: «аппарат разрушен»
        через две секунды после отстрела пусковых мачт.
        """
        # Список деталей — самый дорогой запрос kRPC (объект на каждую
        # деталь). Считаем его не чаще раза в 2 с.
        now = time.monotonic()
        if now - getattr(self, "_parts_checked", 0.0) < 2.0:
            return self._last_part_count
        self._parts_checked = now
        try:
            self._last_part_count = len(self.vessel.parts.all)
        except Exception:
            try:                       # судно могло смениться — перечитаем
                self.vessel = self.sc.active_vessel
                self._last_part_count = len(self.vessel.parts.all)
            except Exception:
                pass                   # держим прошлое значение
        return self._last_part_count

    def __enter__(self) -> "Telemetry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
