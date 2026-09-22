"""Автопилот: план полёта, исполнение и живая таблица.

План строится целиком ДО старта: окно пуска, азимут с поправкой на
вращение планеты, момент перелётного импульса по фазовому углу Луны,
длительность каждого шага. Дальше исполнитель идёт строго по плану,
а таблица показывает для каждого шага:

    действие · ход выполнения · время завершения по часам ПК · отклонение

Отклонение накапливается: если шаг 2 закончился на 2 с раньше, все
следующие прогнозы тоже сдвигаются на −2 с.

План меняется только в крайних случаях — недобор орбиты, промах мимо
Луны, нехватка Δv на посадку. Такие шаги помечаются «добавлено» или
«изменено», а над таблицей появляется отметка «ПЛАН ИЗМЕНЁН».
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from . import mission as ms
from . import term as T
from .analysis import load_genome
from .i18n import L

PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"


# ==========================================================================
@dataclass
class Step:
    key: str
    title: str
    detail: str = ""
    wall: float = 10.0                 # плановая длительность, реальные секунды
    game: float = 0.0                  # плановая длительность, игровые секунды
    dv: float = 0.0
    status: str = PENDING
    mark: str = ""                     # «добавлено» / «изменено»
    note: str = ""
    started: float = 0.0
    finished: float = 0.0
    baseline_end: float = 0.0          # время завершения по исходному плану
    run: Callable | None = None
    progress: Callable | None = None


@dataclass
class FlightPlan:
    target: ms.Target
    steps: list[Step] = field(default_factory=list)
    geometry: ms.LaunchGeometry | None = None
    started: float = 0.0
    changes: int = 0
    finished: bool = False
    success: bool = False
    verdict: str = ""
    # Recovery: when a step fails after its retries, the flight waits for a
    # decision — "retry", "manual" or "rescue". No answer → rescue by itself.
    question: str = ""
    decision: str = ""
    decide_by: float = 0.0

    # --- правки плана ---------------------------------------------------
    def insert_after(self, key: str, step: Step, reason: str) -> None:
        idx = next((i for i, s in enumerate(self.steps) if s.key == key), len(self.steps) - 1)
        step.mark = L("added", "добавлено")
        step.note = reason
        self.steps.insert(idx + 1, step)
        self.changes += 1
        self._rebaseline_from(idx + 1)

    def change(self, key: str, reason: str, title: str | None = None,
               skip: bool = False) -> None:
        for s in self.steps:
            if s.key == key:
                s.mark = L("changed", "изменено")
                s.note = reason
                if title:
                    s.title = title
                if skip:
                    s.status = SKIPPED
                self.changes += 1
                return

    def _rebaseline_from(self, idx: int) -> None:
        """У добавленного шага нет «исходного» времени — оно берётся от
        предыдущего шага, а последующие шаги сдвигаются на его длительность."""
        prev = self.steps[idx - 1].baseline_end if idx > 0 else self.started
        self.steps[idx].baseline_end = prev + self.steps[idx].wall
        for s in self.steps[idx + 1:]:
            s.baseline_end += self.steps[idx].wall

    # --- прогноз --------------------------------------------------------
    def set_baseline(self, start: float) -> None:
        self.started = start
        t = start
        for s in self.steps:
            t += s.wall
            s.baseline_end = t

    def forecast(self) -> list[float]:
        """Прогноз завершения каждого шага с учётом фактических отклонений."""
        now = time.time()
        out = []
        t = self.started
        for s in self.steps:
            if s.status in (DONE, FAILED):
                t = s.finished
            elif s.status == SKIPPED:
                pass
            elif s.status == RUNNING:
                t = max(s.started + s.wall, now)
            else:
                t = t + s.wall
            out.append(t)
        return out


# ==========================================================================
# Построение плана
# ==========================================================================
def _burn_wall(dv: float, accel: float) -> float:
    """Разворот на узел + сам импульс."""
    return 25.0 + dv / max(accel, 0.5)


def _warp_wall(game: float, rate: float) -> float:
    """Сколько реального времени займёт ожидание под варпом."""
    if game <= 0:
        return 2.0
    return min(game, 8.0 + game / rate)


def build(world, target: ms.Target, vessel, report) -> FlightPlan:
    h, m = world.home_body, world.moon_body
    geo = ms.launch_geometry(world, target)
    plan = FlightPlan(target, geometry=geo)
    budget = report.budget if report else ms.budget(world, target)
    legs = {l.key: l for l in budget.legs}
    stages = vessel.stages() if vessel else []
    accel = (stages[-1].thrust_vac / stages[-1].m0) if stages else 5.0
    a0 = (stages[0].thrust_asl / stages[0].m0) if stages else 15.0
    h_leo = ms.leo_altitude(h)
    add = plan.steps.append
    genome = load_genome().get("genome", {}).get("flight", {})
    turn_start = float(genome.get("turn_start_altitude", 1000.0))

    add(Step("check", L("Pre-launch systems check", "Предстартовая проверка систем"),
             L("telemetry, control, stages, game link", "телеметрия, управление, ступени, связь с игрой"),
             wall=4.0))
    if geo.window_wait > 30:
        add(Step("window", L("Waiting for the launch window", "Ожидание окна пуска"),
                 L(f"pad enters {m.name}'s orbital plane, i = {geo.inclination:.1f}°",
                   f"площадка входит в плоскость орбиты {m.name}, i = {geo.inclination:.1f}°"),
                 wall=_warp_wall(geo.window_wait, 1000.0), game=geo.window_wait))
    add(Step("liftoff", L("Liftoff: ignition and vertical climb", "Старт: зажигание и вертикальный подъём"),
             L(f"to H = {turn_start/1000:.1f} km, clear the tower", f"до H = {turn_start/1000:.1f} км, отход от башни"),
             wall=12.0, game=12.0))
    t_turn = max(60.0, legs["ascent"].dv * 0.7 / max(a0 * 1.3, 5.0))
    add(Step("turn", L("Gravity turn", "Гравитационный разворот"),
             L(f"heading {geo.heading:.1f}°, automatic staging, Hα → {h_leo/1000:.0f} km",
               f"азимут {geo.heading:.1f}°, автоматическое отделение ступеней, Hα → {h_leo/1000:.0f} км"),
             wall=t_turn, game=t_turn))
    add(Step("coast", L("Coast to apoapsis, leave the atmosphere", "Выбег до апоцентра, выход из атмосферы"),
             L("fairing jettison, antennas and panels out", "сброс обтекателя, раскрытие антенн и панелей")
             if h.atmosphere_depth else L("free flight to apoapsis", "свободный полёт до апоцентра"),
             wall=45.0, game=60.0))
    v_leo = h.v_circ(h_leo)
    add(Step("circ", L(f"Circularize — low orbit {h_leo/1000:.0f} km",
                       f"Скругление орбиты — выход на НОО {h_leo/1000:.0f} км"),
             L(f"Δv ≈ {v_leo*0.12:.0f} m/s at apoapsis", f"Δv ≈ {v_leo*0.12:.0f} м/с в апоцентре"),
             wall=_burn_wall(v_leo * 0.12, accel), dv=v_leo * 0.12))

    if target.code == "HEO":
        leg = legs["raise"]
        add(Step("raise", L(f"Raise apoapsis to {target.altitude/1000:,.0f} km",
                            f"Подъём апоцентра до {target.altitude/1000:,.0f} км").replace(",", " "),
                 f"Δv = {leg.dv:.0f} " + L("m/s", "м/с"), wall=_burn_wall(leg.dv, accel) + 20, dv=leg.dv))
        _, _, t = ms.hohmann(h, h.radius + h_leo, h.radius + target.altitude)
        add(Step("coast_high", L("Coast to the high apoapsis", "Выбег к апоцентру высокой орбиты"),
                 L(f"{t/60:.0f} min of game time", f"{t/60:.0f} мин игрового времени"), wall=_warp_wall(t, 200.0), game=t))
        leg = legs["circ_high"]
        add(Step("circ_high", L("Circularize the high orbit", "Скругление высокой орбиты"),
                 f"Δv = {leg.dv:.0f} " + L("m/s", "м/с"),
                 wall=_burn_wall(leg.dv, accel), dv=leg.dv))

    if getattr(target, "planet", False):
        _planet_steps(world, target, add, legs, accel, report)
    elif target.body == m.name:
        tli = legs["tli"]
        wait = geo.tli_wait or 0.0
        phase = f"φ = {geo.phase_required:.1f}°" if geo.phase_required is not None else ""
        add(Step("tli_wait", L("Waiting for the transfer window", "Ожидание окна перелёта к Луне"),
                 L(f"{phase}, {wait/60:.0f} min in orbit", f"{phase}, {wait/60:.0f} мин на орбите")
                 + (L(" (estimate)", " (оценка)") if not geo.live else ""),
                 wall=_warp_wall(wait, 50.0) + 15, game=wait))
        add(Step("tli", L(f"Transfer burn to {m.name} (TLI)", f"Перелётный импульс к {m.name} (TLI)"),
                 f"Δv = {tli.dv:.0f} " + L("m/s", "м/с"),
                 wall=_burn_wall(tli.dv, accel), dv=tli.dv))
        _, _, t_tr = ms.transfer_to_moon(h, m, h.radius + h_leo)
        add(Step("coast_soi", L(f"Transfer to {m.name}", f"Перелёт к {m.name}"),
                 L(f"{t_tr/3600:.1f} h of game time to the sphere of influence",
                   f"{t_tr/3600:.1f} ч игрового времени до входа в сферу действия"),
                 wall=_warp_wall(t_tr, 1500.0) + 10, game=t_tr))
        loi = legs["loi"]
        alt = target.altitude if not target.landing else ms.llo_altitude(m)
        add(Step("loi", L(f"Capture burn at {m.name} (LOI)", f"Торможение у {m.name} (LOI)"),
                 L(f"Δv = {loi.dv:.0f} m/s, Hπ = {alt/1000:.0f} km", f"Δv = {loi.dv:.0f} м/с, Hπ = {alt/1000:.0f} км"),
                 wall=_burn_wall(loi.dv, accel) + 40, dv=loi.dv))
        add(Step("orbit_check", L(f"Check the {alt/1000:.0f} km orbit", f"Контроль орбиты {alt/1000:.0f} км"),
                 L("eccentricity, remaining Δv", "эксцентриситет, остаток Δv"), wall=6.0))
        if target.landing:
            land = legs["land"]
            add(Step("deorbit", L("Deorbit", "Сход с орбиты"),
                     L("braking burn, Hπ → near the surface", "импульс торможения, Hπ → у поверхности"),
                     wall=_burn_wall(40.0, accel) + 10, dv=40.0))
            t_fall = m.period_at(alt) / 2
            add(Step("descent_coast", L("Ballistic descent", "Баллистический спуск к грунту"),
                     L(f"{t_fall/60:.0f} min of game time", f"{t_fall/60:.0f} мин игрового времени"),
                     wall=_warp_wall(t_fall, 50.0), game=t_fall))
            add(Step("braking", L("Kill horizontal speed", "Гашение горизонтальной скорости"),
                     f"Δv ≈ {land.dv*0.8:.0f} " + L("m/s", "м/с"), wall=_burn_wall(land.dv * 0.8, accel),
                     dv=land.dv * 0.8))
            add(Step("touchdown", L("Vertical descent and touchdown", "Вертикальный спуск и касание"),
                     L("legs down, touchdown < 3 m/s", "опоры выпущены, v касания < 3 м/с"), wall=60.0, dv=land.dv * 0.2))
    relay = getattr(target, "relay_slot", None)
    if relay:
        slot, total = relay
        add(Step("relay_place", L(f"Place the relay in its slot ({slot + 1}/{total})",
                                  f"Постановка ретранслятора в точку ({slot + 1}/{total})"),
                 L(f"phase {360.0 * slot / total:.0f}° — neighbours see each other",
                   f"фаза {360.0 * slot / total:.0f}° — соседи видят друг друга"), wall=180.0))
    add(Step("done", L("Mission complete", "Миссия выполнена"), target.title, wall=2.0))
    return plan


def _planet_steps(world, target, add, legs, accel, report) -> None:
    """Steps after low home orbit for a planet: window, ejection, cruise with
    trained corrections, capture (aero or engine), working orbit, landing."""
    from . import planets as PL
    from .interplanetary import profile
    vessel = report.vessel if report else None
    heat = bool(vessel and any(pt.resources_max.get("Ablator", 0) > 0 for pt in vessel.parts))
    chutes = bool(vessel and vessel.count("parachute") > 0)
    p = PL.plan(world, target, heat, chutes)
    body = ms.target_body(world, target)
    prof = profile(target.body, p.aero)
    tli, loi = legs["tli"], legs["loi"]
    days = L("days", "сут")
    add(Step("p_window", L(f"Waiting for the window to {target.body}", f"Ожидание окна к {target.body}"),
             L(f"φ = {p.phase:.1f}°, {p.wait / 21600:.0f} {days} of game time",
               f"φ = {p.phase:.1f}°, {p.wait / 21600:.0f} {days} игрового времени"),
             wall=_warp_wall(p.wait, 20000.0) + 20, game=p.wait))
    add(Step("p_eject", L(f"Ejection burn to {target.body}", f"Импульс ухода к {target.body}"),
             L(f"Δv = {tli.dv:.0f} m/s, {prof['eject_lead'] * 100:.0f} % of the burn before the node",
               f"Δv = {tli.dv:.0f} м/с, {prof['eject_lead'] * 100:.0f} % ожога до узла"),
             wall=_burn_wall(tli.dv, accel), dv=tli.dv))
    add(Step("p_cruise", L(f"Cruise to {target.body}", f"Перелёт к {target.body}"),
             L(f"{p.tof / 21600:.0f} {days}; corrections at {prof['corr1'] * 100:.0f} / "
               f"{prof['corr2'] * 100:.0f} / {prof['corr3'] * 100:.1f} % of the way",
               f"{p.tof / 21600:.0f} {days}; коррекции на {prof['corr1'] * 100:.0f} / "
               f"{prof['corr2'] * 100:.0f} / {prof['corr3'] * 100:.1f} % пути"),
             wall=_warp_wall(p.tof, 60000.0) + 90, game=p.tof))
    if p.aero:
        add(Step("p_capture", L(f"Aerocapture at {target.body}", f"Аэрозахват у {target.body}"),
                 L(f"periapsis at {prof['aero_depth'] * 100:.0f} % of the atmosphere, heat shield forward",
                   f"перицентр на {prof['aero_depth'] * 100:.0f} % толщи атмосферы, щитом вперёд"),
                 wall=240.0, dv=loi.dv))
    else:
        add(Step("p_capture", L(f"Capture burn at {target.body}", f"Торможение у {target.body}"),
                 f"Δv = {loi.dv:.0f} " + L("m/s", "м/с"), wall=_burn_wall(loi.dv, accel) + 60, dv=loi.dv))
    work = ms.work_altitude(world, target)
    add(Step("p_orbit", L(f"Working orbit {work / 1000:.0f} km", f"Рабочая орбита {work / 1000:.0f} км"),
             L("lower and circularize", "снижение и скругление"), wall=120.0))
    if target.landing:
        land = legs["land"]
        if body.atmosphere_depth > 0:
            add(Step("c_deorbit", L("Deorbit into the atmosphere", "Сход с орбиты в атмосферу"),
                     L("periapsis at a quarter of the atmosphere", "перицентр на четверти толщи атмосферы"),
                     wall=80.0, dv=land.dv * 0.3))
            add(Step("c_entry", L("Entry and parachutes", "Вход в атмосферу и парашюты"),
                     L("heat shield forward, canopies below 450 m/s", "щитом вперёд, купола ниже 450 м/с"),
                     wall=240.0))
            add(Step("c_touchdown", L("Touchdown", "Касание"),
                     L("engine trims the last 300 m if the air is thin", "двигатель доводит последние 300 м, если воздух разрежен"),
                     wall=120.0, dv=land.dv * 0.2))
        else:
            add(Step("deorbit", L("Deorbit", "Сход с орбиты"),
                     L("braking burn, Hπ → near the surface", "импульс торможения, Hπ → у поверхности"),
                     wall=_burn_wall(40.0, accel) + 10, dv=40.0))
            add(Step("descent_coast", L("Ballistic descent", "Баллистический спуск к грунту"), "",
                     wall=_warp_wall(body.period_at(work) / 2, 50.0)))
            add(Step("braking", L("Kill horizontal speed", "Гашение горизонтальной скорости"),
                     f"Δv ≈ {land.dv * 0.8:.0f} " + L("m/s", "м/с"), wall=_burn_wall(land.dv * 0.8, accel)))
            add(Step("touchdown", L("Vertical descent and touchdown", "Вертикальный спуск и касание"),
                     L("legs down, touchdown < 3 m/s", "опоры выпущены, v касания < 3 м/с"), wall=60.0))


# ==========================================================================
class LogTail(logging.Handler):
    """Ловит сообщения пилотов для строки журнала под таблицей."""

    def __init__(self, size: int = 6):
        super().__init__(logging.INFO)
        self.lines = collections.deque(maxlen=size)

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if record.levelno >= logging.INFO and not msg.startswith("Прейскурант"):
            stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            color = T.WARN if record.levelno >= logging.WARNING else T.MUTED
            self.lines.append(f"{color}{stamp}  {msg}{T.RESET}")


class Executor:
    def __init__(self, world, plan: FlightPlan, vessel, report):
        self.world = world
        self.plan = plan
        self.vessel = vessel
        self.report = report
        self.abort = threading.Event()
        self.thread: threading.Thread | None = None
        self.error = ""
        self.ctx: dict = {}
        self.answer = threading.Event()
        self.tail = LogTail()
        logging.getLogger("kia").addHandler(self.tail)

    # --- подготовка пилотов -----------------------------------------------
    def _setup(self) -> bool:
        from kia_core.environment.control import ShipControl
        from kia_core.environment.telemetry import Telemetry
        from kia_core.learning.genome import FlightGenome
        from kia_core.learning.rewards import RewardSystem
        from kia_core.mission_spec import MissionSpec
        from kia_core.pilot.ascent import AscentAutopilot
        from kia_core.pilot.maneuver import ManeuverExecutor
        from kia_core.pilot.transfer import TransferPlanner

        conn = self.world.connection
        t = self.plan.target
        h, m = self.world.home_body, self.world.moon_body
        h_leo = ms.leo_altitude(h)
        alt = ms.work_altitude(self.world, t)
        objectives = ["land"] if t.landing else ["orbit"]
        spec = MissionSpec(title=t.title, target_body=t.body, home_body=h.name,
                           park_orbit_altitude=h_leo,
                           target_orbit_altitude=alt,
                           objectives=objectives, collect_science=False)
        g = FlightGenome.from_dict(load_genome().get("genome", {}).get("flight", {}))
        g.target_apoapsis = h_leo
        g.ascent_heading = self.plan.geometry.heading if self.plan.geometry else 90.0
        g.mun_periapsis_target = alt
        tel = Telemetry(conn)
        ctl = ShipControl(conn, tel)
        man = ManeuverExecutor(conn, tel, ctl)
        rewards = RewardSystem(spec=spec)
        self.ctx.update(conn=conn, tel=tel, ctl=ctl, man=man, rewards=rewards, g=g,
                        spec=spec, h_leo=h_leo, alt=alt)
        self.ctx["ascent"] = AscentAutopilot(conn, tel, ctl, man, g, rewards,
                                             target_apoapsis=h_leo)
        self.ctx["home"] = h.name
        if getattr(t, "planet", False):
            from . import planets as PL
            from .interplanetary import ChuteLanding, PlanetFlight
            tr = TransferPlanner(conn, tel, ctl, man, g, rewards, target=t.body,
                                 target_periapsis=alt)
            tr._math_brain = False
            self.ctx["transfer"] = tr
            v = self.vessel
            heat = bool(v and any(pt.resources_max.get("Ablator", 0) > 0 for pt in v.parts))
            aero = PL.plan(self.world, t, heat, bool(v and v.count("parachute"))).aero
            self.ctx["flight"] = PlanetFlight(self.ctx, t.body, alt, aero)
            self.ctx["chute"] = ChuteLanding(self.ctx, t.body)
        elif t.body == m.name:
            tr = TransferPlanner(conn, tel, ctl, man, g, rewards, target=m.name,
                                 target_periapsis=alt)
            tr._math_brain = False          # без дообучения: только готовые веса
            self.ctx["transfer"] = tr
        return True

    def _bind(self) -> None:
        c = self.ctx
        sc = lambda: c["conn"].space_center          # noqa: E731
        tel = lambda k: c["tel"].get(k)               # noqa: E731
        geo = self.plan.geometry
        for s in self.plan.steps:
            k = s.key
            if k == "check":
                s.run = self._check
            elif k == "window":
                ut0 = sc().ut
                target_ut = ut0 + geo.window_wait
                s.run = lambda target_ut=target_ut: (c["ctl"].warp_to(target_ut) or True)
                s.progress = lambda ut0=ut0, t=target_ut: (tel("ut") - ut0) / max(t - ut0, 1)
            elif k == "liftoff":
                s.run = c["ascent"].launch
                ts = c["g"].turn_start_altitude
                s.progress = lambda ts=ts: tel("altitude") / max(ts, 1)
            elif k == "turn":
                s.run = c["ascent"].gravity_turn
                s.progress = lambda: tel("apoapsis") / c["h_leo"]
            elif k == "coast":
                s.run = c["ascent"].coast_to_space
                depth = self.world.home_body.atmosphere_depth or c["h_leo"]
                s.progress = lambda d=depth: tel("altitude") / d
            elif k == "circ":
                s.run = self._circularize
                s.progress = lambda: max(0.0, tel("periapsis")) / c["h_leo"]
            elif k == "raise":
                alt = self.plan.target.altitude
                s.run = lambda alt=alt: c["man"].change_apoapsis(alt, c["g"].node_execute_lead)
                s.progress = lambda alt=alt: (tel("apoapsis") - c["h_leo"]) / (alt - c["h_leo"])
            elif k == "coast_high":
                s.run = self._coast_to_apoapsis
            elif k == "circ_high":
                s.run = lambda: c["man"].circularize_at_apoapsis(c["g"].circularization_lead)
                alt = self.plan.target.altitude
                s.progress = lambda alt=alt: max(0.0, tel("periapsis")) / alt
            elif k == "tli_wait":
                s.run = self._tli_wait
            elif k == "tli":
                s.run = self._tli
            elif k == "coast_soi":
                s.run = c["transfer"].coast_to_soi
            elif k == "loi":
                s.run = c["transfer"].capture
            elif k == "orbit_check":
                s.run = self._orbit_check
            elif k == "deorbit":
                s.run = lambda: self._lander().deorbit()
            elif k == "descent_coast":
                s.run = lambda: self._lander().coast_to_surface()
            elif k == "braking":
                s.run = lambda: self._lander().kill_horizontal()
            elif k == "touchdown":
                s.run = lambda: self._lander().descend()
            elif k == "relay_place":
                slot, total = self.plan.target.relay_slot

                def place(slot=slot, total=total):
                    from kia_core.pilot.constellation import ConstellationPilot
                    pilot = ConstellationPilot(c["conn"], c["ctl"], c["man"], c["tel"])
                    return pilot.place(360.0 * slot / total)
                s.run = place
            elif k == "p_window":
                s.run = lambda: c["flight"].window()
            elif k == "p_eject":
                s.run = lambda: c["flight"].eject()
            elif k == "p_cruise":
                s.run = lambda: c["flight"].cruise()
            elif k == "p_capture":
                s.run = lambda: c["flight"].capture()
            elif k == "p_orbit":
                s.run = lambda: c["flight"].working_orbit()
            elif k == "c_deorbit":
                s.run = lambda: c["chute"].deorbit()
            elif k == "c_entry":
                s.run = lambda: c["chute"].entry()
            elif k == "c_touchdown":
                s.run = lambda: c["chute"].touchdown()
            elif k == "done":
                s.run = lambda: True
            if s.progress is None:
                s.progress = lambda s=s: (time.time() - s.started) / max(s.wall, 1)

    # --- шаги с логикой ---------------------------------------------------
    def _check(self) -> bool:
        tel = self.ctx["tel"]
        snap = tel.snapshot()
        if snap.situation not in ("pre_launch", "landed"):
            self.error = L(f"the craft is not on the pad ({snap.situation})", f"аппарат не на стартовом столе ({snap.situation})")
            return False
        return True

    def _circularize(self) -> bool:
        ok = self.ctx["ascent"].circularize()
        orbit = self.ctx["tel"].vessel.orbit
        h = self.world.home_body
        floor = max(h.atmosphere_depth, 0.0) + 2_000.0
        if orbit.periapsis_altitude < floor:
            self.plan.insert_after("circ", Step(
                "circ_fix", L("Orbit fix: raise periapsis", "Довыведение: подъём перицентра"),
                L(f"Hπ = {orbit.periapsis_altitude/1000:.0f} km is below the atmosphere",
                  f"Hπ = {orbit.periapsis_altitude/1000:.0f} км ниже границы атмосферы"),
                wall=40.0, run=lambda: self.ctx["man"].circularize_at_apoapsis(0.5)),
                reason=L("the orbit did not close above the atmosphere",
                         "орбита не замкнулась выше атмосферы"))
            self._bind_added()
            return True
        return ok

    def _coast_to_apoapsis(self) -> bool:
        v = self.ctx["tel"].vessel
        self.ctx["ctl"].warp_to(self.ctx["conn"].space_center.ut + v.orbit.time_to_apoapsis - 60)
        return True

    def _tli_wait(self) -> bool:
        step = next(s for s in self.plan.steps if s.key == "tli_wait")
        sc = self.ctx["conn"].space_center
        planned = sc.ut + (self.plan.geometry.tli_wait or 0.0) if step.started else None
        node = self.ctx["transfer"].plan_transfer()
        if node is None:
            self.error = L("could not build the transfer node", "не удалось построить узел перелёта")
            return False
        self.ctx["node"] = node
        try:
            shift = node.ut - (planned or node.ut)
            if abs(shift) > 180:
                self.plan.change("tli_wait", L("window refined from the orbit: ", "окно уточнено по орбите: ")
                                 + f"{'+' if shift > 0 else '−'}{abs(shift)/60:.0f} " + L("min", "мин"))
        except Exception:
            pass
        return True

    def _tli(self) -> bool:
        node = self.ctx.get("node")
        if node is None:
            return False
        ok = self.ctx["transfer"].execute_transfer(node)
        try:
            orbit = self.ctx["tel"].vessel.orbit
            nxt = orbit.next_orbit
            m = self.world.moon
            hit = nxt is not None and nxt.body.name == m
            pe_ok = hit and 0 < nxt.periapsis_altitude < self.ctx["alt"] * 4 + 50_000
        except Exception:
            hit, pe_ok = True, True
        if not pe_ok:
            reason = (L("no encounter with the Moon", "встречи с Луной нет") if not hit
                      else L("periapsis at the Moon is far from plan", "перицентр у Луны далёк от плана"))
            self.plan.insert_after("tli", Step(
                "mcc", L("Course correction", "Коррекция курса"),
                L("refine the approach, Δv ≤ 150 m/s", "уточнение точки подлёта, Δv ≤ 150 м/с"),
                wall=60.0, run=self.ctx["transfer"].correct_course), reason=reason)
            self._bind_added()
        return ok

    def _orbit_check(self) -> bool:
        if not self.plan.target.landing:
            return True
        from .craft import from_krpc
        try:
            v = from_krpc(self.ctx["conn"])
            left = v.dv_vac_total
        except Exception:
            return True
        need = ms.landing_dv(self.world.moon_body, self.ctx["alt"])
        if left < need:
            reason = L(f"Δv left {left:.0f} < {need:.0f} m/s — landing cancelled, staying in orbit",
                       f"остаток Δv {left:.0f} < {need:.0f} м/с — посадка отменена, аппарат на орбите")
            for key in ("deorbit", "descent_coast", "braking", "touchdown"):
                self.plan.change(key, reason, skip=True)
            self.plan.changes -= 3              # одна правка, а не четыре
        return True

    def _lander(self):
        if "lander" not in self.ctx:
            from kia_core.pilot.landing import LandingPilot
            c = self.ctx
            c["lander"] = LandingPilot(c["conn"], c["tel"], c["ctl"], c["man"],
                                       c["rewards"], target_name=self.plan.target.body)
        return self.ctx["lander"]

    def _bind_added(self) -> None:
        for s in self.plan.steps:
            if s.progress is None:
                s.progress = lambda s=s: (time.time() - s.started) / max(s.wall, 1)

    # --- цикл -------------------------------------------------------------
    def start(self) -> None:
        from . import rpcprofile
        if not getattr(rpcprofile, "_started", False):
            rpcprofile._started = True
            rpcprofile.start(lambda: self.world.connection.space_center)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self._setup()
            self._bind()
            self.plan.set_baseline(time.time())
        except Exception as exc:
            self.error = L(f"setup: {exc}", f"подготовка: {exc}")
            self.plan.finished = True
            return
        i = 0
        while i < len(self.plan.steps):
            s = self.plan.steps[i]
            i += 1
            if s.status == SKIPPED:
                continue
            if self.abort.is_set():
                s.status = SKIPPED
                s.note = s.note or L("flight aborted by the operator", "полёт прерван оператором")
                continue
            ok = self._attempt(s)
            if not ok and s.key not in ("mcc", "circ_fix"):
                choice = self._ask_recovery(s)
                if choice == "retry":
                    i -= 1                      # the same step once more
                    continue
                for rest in self.plan.steps[i:]:
                    if rest.status == PENDING:
                        rest.status = SKIPPED
                if choice == "manual":
                    self._hand_over()
                    self.plan.verdict = L("control handed to the pilot", "управление передано пилоту")
                    self.plan.finished = True
                    return
                self._rescue(s)
                break
        self.plan.success = all(s.status in (DONE, SKIPPED) for s in self.plan.steps) \
            and self.plan.steps[-1].status == DONE
        if self.plan.success:
            self.plan.verdict = L("target reached", "цель достигнута")
        try:
            self.ctx["ctl"].full_stop()
        except Exception:
            pass
        self.plan.finished = True

    # --- recovery -----------------------------------------------------------
    RETRIES = 2
    NO_RETRY = {"check", "liftoff", "turn", "coast", "c_entry", "c_touchdown", "touchdown",
                "descent_coast", "done"}      # continuous phases: a repeat makes no sense

    def _attempt(self, s: Step) -> bool:
        tries = 1 if s.key in self.NO_RETRY else 1 + self.RETRIES
        ok = False
        for n in range(tries):
            if self.abort.is_set():
                break
            if n:
                s.mark = L(f"retry {n}/{self.RETRIES}", f"повтор {n}/{self.RETRIES}")
                s.note = self.error
                logging.getLogger("kia").warning("retry %s: %s", s.key, self.error)
                try:
                    self.ctx["ctl"].stop_warp()
                    self.ctx["ctl"].set_throttle(0.0)
                except Exception:
                    pass
                time.sleep(2.0)
            s.status = RUNNING
            s.started = time.time()
            try:
                ok = bool(s.run()) if s.run else True
            except Exception as exc:
                ok = False
                self.error = f"{s.title}: {exc}"
                logging.getLogger("kia").error("%s\n%s", exc, traceback.format_exc())
            s.finished = time.time()
            if ok:
                break
        s.status = DONE if ok else FAILED
        return ok

    def _ask_recovery(self, s: Step, wait: float = 60.0) -> str:
        if self.abort.is_set():
            return "rescue"
        self.plan.question = L(f"Step “{s.title}” failed" + (f" ({self.error})" if self.error else "") + ".",
                               f"Шаг «{s.title}» не удался" + (f" ({self.error})" if self.error else "") + ".")
        self.plan.decision = ""
        self.plan.decide_by = time.time() + wait
        self.answer.clear()
        self.answer.wait(wait)
        choice = self.plan.decision or "rescue"
        self.plan.question = ""
        return choice

    def decide(self, choice: str) -> None:
        self.plan.decision = choice
        self.answer.set()

    def _hand_over(self) -> None:
        c = self.ctx
        try:
            c["ctl"].set_throttle(0.0)
            c["ctl"].stop_warp()
            c["ctl"].disengage_autopilot(hold=False)
            c["ctl"].set_sas(True, "stability_assist")
        except Exception:
            pass

    def _rescue(self, failed: Step) -> None:
        """Bring the craft into a safe state: a stable orbit, or a soft landing."""
        c = self.ctx
        step = Step("rescue", L("Rescue the craft", "Спасение аппарата"), wall=120.0, mark=L("added", "добавлено"))
        self.plan.steps.append(step)
        self.plan.changes += 1
        step.status, step.started = RUNNING, time.time()
        ok, how = False, ""
        try:
            v = c["tel"].vessel
            body = v.orbit.body
            atmo = body.has_atmosphere
            pe, ap = v.orbit.periapsis_altitude, v.orbit.apoapsis_altitude
            floor = (body.atmosphere_depth if atmo else 0.0) + 5_000.0
            c["ctl"].stop_warp()
            sit = str(v.situation).split(".")[-1]
            if sit in ("pre_launch", "landed", "splashed"):
                c["ctl"].set_throttle(0.0)
                how, ok = L("on the ground, engines off", "на земле, двигатели выключены"), True
            elif pe > floor:
                how, ok = L("already in a stable orbit", "уже на устойчивой орбите"), True
            elif ap > floor and v.orbit.time_to_apoapsis > 60:
                how = L("circularize at apoapsis", "скругление орбиты в апоцентре")
                ok = bool(c["man"].circularize_at_apoapsis(0.5))
            elif atmo:
                how = L("parachutes, retrograde", "парашюты, ретроград")
                c["ctl"].set_throttle(0.0)
                c["ctl"].set_sas(True, "retrograde")
                while v.flight(body.reference_frame).mean_altitude > 5_000 and not self.abort.is_set():
                    time.sleep(1.0)
                c["ctl"].deploy_parachutes()
                ok = True
            else:
                how = L("powered landing", "посадка на двигателях")
                lander = self._lander()
                ok = bool(lander.kill_horizontal()) and bool(lander.descend())
        except Exception as exc:
            how = how or str(exc)
        step.detail = how
        step.finished = time.time()
        step.status = DONE if ok else FAILED
        base = self.error or L(f"step “{failed.title}” failed", f"шаг «{failed.title}» не выполнен")
        self.plan.verdict = base + " — " + (L("craft saved: ", "аппарат спасён: ") if ok
                                            else L("rescue failed: ", "спасти не удалось: ")) + how

    def stop(self) -> None:
        self.abort.set()
        self.answer.set()
        try:
            self.ctx["ctl"].set_throttle(0.0)
        except Exception:
            pass

    def telemetry_line(self) -> str:
        tel = self.ctx.get("tel")
        if tel is None:
            return ""
        try:
            g = tel.get
            km, ms_ = L("km", "км"), L("m/s", "м/с")
            return (f"H {g('altitude')/1000:8.1f} {km}   Hα {g('apoapsis')/1000:8.1f} {km}   "
                    f"Hπ {g('periapsis')/1000:8.1f} {km}   v {g('speed'):7.0f} {ms_}   "
                    f"TWR {tel.current_twr():4.2f}   T+ {_clock(g('mission_time'))}")
        except Exception:
            return ""


# ==========================================================================
# Отрисовка
# ==========================================================================
def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _delta(seconds: float) -> str:
    s = int(round(seconds))
    if abs(s) < 1:
        return f"{T.MUTED}±0{L('s', 'с')}{T.RESET}"
    color = T.BAD if s > 0 else T.OK
    sign = "+" if s > 0 else "−"
    if abs(s) >= 3600:
        text = f"{abs(s)//3600}{L('h', 'ч')}{abs(s)%3600//60:02d}{L('m', 'м')}"
    elif abs(s) >= 60:
        text = f"{abs(s)//60}{L('m', 'м')}{abs(s)%60:02d}{L('s', 'с')}"
    else:
        text = f"{abs(s)}{L('s', 'с')}"
    return f"{color}{sign}{text}{T.RESET}"


def render(plan: FlightPlan, vessel_name: str, telemetry: str = "",
           log_lines: list[str] | None = None, footer: str = "") -> list[str]:
    w, h = T.size()
    now = time.time()
    fc = plan.forecast()
    out = []
    title = f"{T.BOLD}{T.SILVER}ASTRAUTOMA{T.RESET}{T.STEEL} · {L('AUTOPILOT', 'АВТОПИЛОТ')}{T.RESET}"
    right = datetime.now().strftime("%H:%M:%S")
    info = f"{T.SKY}{plan.target.code}{T.RESET} {plan.target.title}   {T.MUTED}{L('craft:', 'аппарат:')}{T.RESET} {vessel_name}"
    out.append(" " + T.pad(title + "   " + info, w - 12) + T.WHITE + right + T.RESET)
    if plan.changes:
        out.append(" " + T.bg(90, 60, 0) + T.WHITE + T.BOLD
                   + L(f" PLAN CHANGED · edits: {plan.changes} ", f" ПЛАН ИЗМЕНЁН · правок: {plan.changes} ")
                   + T.RESET + T.MUTED
                   + L("  steps marked “added” / “changed”", "  шаги с пометкой «добавлено» / «изменено»") + T.RESET)
    else:
        out.append(" " + T.MUTED + L("plan built before launch: launch window, heading, phase angle, "
                                     "planet rotation and the Moon's motion included",
                                     "план построен до старта: окно пуска, азимут, фазовый угол, "
                                     "вращение планеты и движение Луны учтены") + T.RESET)
    geo = plan.geometry
    if geo is not None:
        bits = [L(f"heading {geo.heading:.1f}°", f"азимут {geo.heading:.1f}°")]
        if geo.phase_required is not None:
            bits.append(f"φ TLI {geo.phase_required:.1f}°")
        if geo.window_wait > 30:
            bits.append(L(f"launch window in {_clock(geo.window_wait)}", f"окно пуска через {_clock(geo.window_wait)}"))
        out.append(" " + T.STEEL + "  ·  ".join(bits) + T.RESET)
    out.append("")

    col_eta, col_dt, col_bar = 10, 9, 18
    col_title = max(30, w - col_eta - col_dt - col_bar - 16)
    out.append(" " + T.MUTED + T.pad(" #", 4) + T.pad(L("ACTION", "ДЕЙСТВИЕ"), col_title)
               + T.pad(L("PROGRESS", "ВЫПОЛНЕНИЕ"), col_bar + 6) + T.pad(L("ARRIVAL", "ПРИБЫТИЕ"), col_eta)
               + T.pad("Δt", col_dt, "right") + T.RESET)
    out.append(" " + T.STEEL + "─" * (w - 3) + T.RESET)

    for i, (s, eta) in enumerate(zip(plan.steps, fc), 1):
        if s.status == DONE:
            icon, color, frac = f"{T.OK}✓", T.SILVER, 1.0
        elif s.status == RUNNING:
            spin = "◐◓◑◒"[int(now * 4) % 4]
            icon, color = f"{T.FLAME}{spin}", T.WHITE + T.BOLD
            try:
                frac = max(0.0, min(0.99, float(s.progress())))
            except Exception:
                frac = min(0.99, (now - s.started) / max(s.wall, 1))
        elif s.status == FAILED:
            icon, color, frac = f"{T.BAD}✗", T.BAD, 0.0
        elif s.status == SKIPPED:
            icon, color, frac = f"{T.MUTED}–", T.MUTED, 0.0
        else:
            icon, color, frac = f"{T.MUTED}○", T.STEEL, 0.0

        mark = ""
        if s.mark:
            mark = f" {T.bg(90, 60, 0)}{T.WHITE} {s.mark} {T.RESET}"
        name = f"{color}{s.title}{T.RESET}{mark}"
        if s.status == SKIPPED:
            bar_txt = T.MUTED + T.pad(L("skipped", "пропущено"), col_bar + 6) + T.RESET
            eta_txt, dt_txt = T.pad("—", col_eta), ""
        else:
            bar_color = T.OK if s.status == DONE else (T.BAD if s.status == FAILED else T.SKY)
            bar_txt = T.bar(frac, col_bar, bar_color) + f" {int(frac*100):3d}%  "
            eta_txt = T.pad(datetime.fromtimestamp(eta).strftime("%H:%M:%S"), col_eta)
            dt_txt = _delta(eta - s.baseline_end) if s.baseline_end else ""
        out.append(f" {icon}{T.RESET} {T.MUTED}{i:>2}{T.RESET} " + T.pad(T.cut(name, col_title - 1), col_title)
                   + bar_txt + eta_txt + T.pad(dt_txt, col_dt, "right"))
        sub = s.note if s.note else s.detail
        if sub and s.status in (RUNNING, PENDING) or s.mark:
            c = T.WARN if s.mark else T.MUTED
            out.append("       " + c + T.cut(sub, col_title + col_bar) + T.RESET)

    out.append(" " + T.STEEL + "─" * (w - 3) + T.RESET)
    if telemetry:
        out.append(" " + T.SKY + telemetry + T.RESET)
    if log_lines:
        out.append("")
        for line in list(log_lines)[-5:]:
            out.append("   " + line)
    if plan.finished:
        out.append("")
        c = T.OK if plan.success else T.BAD
        out.append(" " + c + T.BOLD + ("✓ " if plan.success else "✗ ")
                   + plan.verdict.upper() + T.RESET)
    if footer:
        pad_lines = h - 2 - len(out)
        out.extend([""] * max(0, pad_lines))
        out.append(" " + T.MUTED + footer + T.RESET)
    return out


# ==========================================================================
# Демонстрация таблицы без игры (для проверки интерфейса)
# ==========================================================================
def demo_run(plan: FlightPlan, speed: float = 1.0) -> None:
    """Прогоняет план с искусственными длительностями: шаг идёт от 0.6 до 1.3
    своего плана, на скруглении добавляется довыведение."""
    import random
    rnd = random.Random(7)
    for s in plan.steps:
        s.wall = s.wall / speed
    plan.set_baseline(time.time())

    def worker():
        i = 0
        while i < len(plan.steps):
            s = plan.steps[i]
            i += 1
            if s.status == SKIPPED:
                continue
            s.status = RUNNING
            s.started = time.time()
            dur = s.wall * rnd.uniform(0.6, 1.3)
            s.progress = lambda s=s, d=dur: (time.time() - s.started) / d
            time.sleep(dur)
            s.finished = time.time()
            s.status = DONE
            if s.key == "circ" and not any(x.key == "circ_fix" for x in plan.steps):
                plan.insert_after("circ", Step("circ_fix", L("Orbit fix: raise periapsis",
                                                             "Довыведение: подъём перицентра"),
                                               L("Hπ below the atmosphere", "Hπ ниже границы атмосферы"),
                                               wall=8.0 / speed),
                                  reason=L("the orbit did not close above the atmosphere",
                                           "орбита не замкнулась выше атмосферы"))
        plan.success = True
        plan.verdict = L("target reached (demo)", "цель достигнута (демонстрация)")
        plan.finished = True

    threading.Thread(target=worker, daemon=True).start()
