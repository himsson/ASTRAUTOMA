"""Солнечная система KSP для межпланетного тренажёра.

Параметры тел сняты с живой игры через kRPC (data/solar_system.json):
μ, радиусы, сферы действия, атмосферы и орбитальные элементы с эпохой.
Поэтому тренажёр считает те же положения планет, что и игра, в любой
момент игрового времени.

Всё векторизовано на numpy: тренажёр гоняет тысячи миссий за раз, и
решатель Ламберта работает сразу по всей пачке.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DATA_DIR

SOLAR_FILE = DATA_DIR / "solar_system.json"

HOME = "Kerbin"
# Цели межпланетной программы: все планеты стоковой системы, кроме дома
PLANETS = ("Moho", "Eve", "Duna", "Dres", "Jool", "Eeloo")
# Аналоги из реальной системы — для подписей и для команд оператора
REAL_NAMES = {"Moho": "Меркурий", "Eve": "Венера", "Duna": "Марс",
              "Dres": "Дрес", "Jool": "Юпитер", "Eeloo": "Плутон"}
GAS_GIANTS = {"Jool"}

# Высшие точки рельефа (м): kRPC их не отдаёт, значения из игры 1.12
TERRAIN = {"Moho": 6_817.0, "Eve": 7_526.0, "Duna": 8_264.0, "Dres": 5_700.0,
           "Jool": 0.0, "Eeloo": 3_900.0, "Kerbin": 6_767.0}


@dataclass
class Body:
    name: str
    mu: float
    radius: float
    soi: float
    atmosphere: float
    rotation_period: float
    parent: str | None = None
    sma: float = 0.0
    ecc: float = 0.0
    inc: float = 0.0
    lan: float = 0.0
    argp: float = 0.0
    mean_anomaly_at_epoch: float = 0.0
    epoch: float = 0.0
    period: float = 0.0
    sea_level_pressure: float = 0.0
    satellites: list[str] = field(default_factory=list)

    @property
    def g0(self) -> float:
        return self.mu / self.radius ** 2

    @property
    def terrain(self) -> float:
        return TERRAIN.get(self.name, self.radius * 0.03)

    def v_circ(self, altitude):
        return np.sqrt(self.mu / (self.radius + altitude))

    # Целевые орбиты — те же правила, что в ASTRAUTOMA
    def low_orbit(self) -> float:
        if self.atmosphere > 0:
            return math.ceil((self.atmosphere + 10_000.0) / 5_000.0) * 5_000.0
        alt = max(self.terrain + 10_000.0, self.radius * 0.1)
        return math.ceil(alt / 5_000.0) * 5_000.0

    def high_orbit(self) -> float:
        return round((self.soi * 0.2) / 10_000.0) * 10_000.0


class SolarSystem:
    def __init__(self, path: Path | None = None):
        data = json.loads(Path(path or SOLAR_FILE).read_text(encoding="utf-8"))
        self.bodies: dict[str, Body] = {}
        for name, d in data["bodies"].items():
            self.bodies[name] = Body(
                name=name, mu=d["mu"], radius=d["radius"], soi=d.get("soi") or math.inf,
                atmosphere=d.get("atmosphere_depth", 0.0),
                rotation_period=d.get("rotation_period", 0.0),
                parent=d.get("parent"), sma=d.get("sma", 0.0), ecc=d.get("ecc", 0.0),
                inc=d.get("inc", 0.0), lan=d.get("lan", 0.0), argp=d.get("argp", 0.0),
                mean_anomaly_at_epoch=d.get("mean_anomaly_at_epoch", 0.0),
                epoch=d.get("epoch", 0.0), period=d.get("period", 0.0),
                sea_level_pressure=d.get("sea_level_pressure", 0.0),
                satellites=d.get("satellites", []))
        self.sun = self.bodies["Sun"]

    def __getitem__(self, name: str) -> Body:
        return self.bodies[name]

    # ------------------------------------------------------------------
    def state(self, name: str, ut):
        """Положение и скорость тела относительно Солнца (м, м/с).

        ut — скаляр или массив; результат формы (..., 3). Система отсчёта
        правая, ось z — к северу эклиптики.
        """
        b = self.bodies[name]
        ut = np.asarray(ut, dtype=np.float64)
        n = 2 * math.pi / b.period
        M = b.mean_anomaly_at_epoch + n * (ut - b.epoch)
        E = M.copy() if M.ndim else np.array(M)
        for _ in range(12):                 # эксцентриситеты малы (Эелу 0.26)
            E = E - (E - b.ecc * np.sin(E) - M) / (1 - b.ecc * np.cos(E))
        cos_e, sin_e = np.cos(E), np.sin(E)
        a, e = b.sma, b.ecc
        x = a * (cos_e - e)
        y = a * math.sqrt(1 - e * e) * sin_e
        r = a * (1 - e * cos_e)
        vfac = math.sqrt(self.sun.mu * a) / r
        vx = -vfac * sin_e
        vy = vfac * math.sqrt(1 - e * e) * cos_e
        rot = _rotation(b.lan, b.inc, b.argp)
        pos = np.stack([x, y, np.zeros_like(x)], axis=-1) @ rot.T
        vel = np.stack([vx, vy, np.zeros_like(vx)], axis=-1) @ rot.T
        return pos, vel

    def phase_angle(self, target: str, ut):
        """Угол цели впереди Кербина по ходу движения, рад в [0, 2π)."""
        rk, _ = self.state(HOME, ut)
        rt, _ = self.state(target, ut)
        ang = np.arctan2(rt[..., 1], rt[..., 0]) - np.arctan2(rk[..., 1], rk[..., 0])
        return np.mod(ang, 2 * math.pi)

    def synodic(self, target: str) -> float:
        tk, tt = self.bodies[HOME].period, self.bodies[target].period
        return abs(1.0 / (1.0 / tk - 1.0 / tt))

    def hohmann(self, target: str) -> tuple[float, float, float]:
        """(Δv вылета с НОО, v∞ прилёта, время перелёта) по Гоману — ориентир."""
        k, t = self.bodies[HOME], self.bodies[target]
        mu = self.sun.mu
        r1, r2 = k.sma, t.sma
        a = (r1 + r2) / 2
        v_inf_dep = abs(math.sqrt(mu * (2 / r1 - 1 / a)) - math.sqrt(mu / r1))
        v_inf_arr = abs(math.sqrt(mu / r2) - math.sqrt(mu * (2 / r2 - 1 / a)))
        r_park = k.radius + 80_000.0
        dv = math.sqrt(v_inf_dep ** 2 + 2 * k.mu / r_park) - math.sqrt(k.mu / r_park)
        return dv, v_inf_arr, math.pi * math.sqrt(a ** 3 / mu)


def _rotation(lan: float, inc: float, argp: float) -> np.ndarray:
    cO, sO = math.cos(lan), math.sin(lan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])


# ==========================================================================
# Ламберт пачкой
# ==========================================================================
def _stumpff(z):
    c = np.empty_like(z)
    s = np.empty_like(z)
    pos, neg, zero = z > 1e-6, z < -1e-6, np.abs(z) <= 1e-6
    sz = np.sqrt(z[pos])
    c[pos] = (1 - np.cos(sz)) / z[pos]
    s[pos] = (sz - np.sin(sz)) / sz ** 3
    sn = np.sqrt(-z[neg])
    c[neg] = (np.cosh(sn) - 1) / (-z[neg])
    s[neg] = (np.sinh(sn) - sn) / sn ** 3
    c[zero] = 0.5
    s[zero] = 1.0 / 6.0
    return c, s


def lambert(r1, r2, tof, mu: float, iterations: int = 64):
    """Задача Ламберта (универсальные переменные, один виток, прямое движение).

    r1, r2: (N, 3); tof: (N,). Возвращает v1, v2 (N, 3) и маску решённых.
    Бисекция по z устойчивее Ньютона: функция монотонна, а пачке нужна
    гарантия сходимости для каждого элемента, а не в среднем.
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    tof = np.asarray(tof, dtype=np.float64)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    cos_dt = np.clip(np.sum(r1 * r2, axis=-1) / (n1 * n2), -1.0, 1.0)
    dtheta = np.arccos(cos_dt)
    cross_z = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    dtheta = np.where(cross_z < 0, 2 * math.pi - dtheta, dtheta)
    A = np.sin(dtheta) * np.sqrt(n1 * n2 / np.maximum(1 - np.cos(dtheta), 1e-12))
    sq_mu = math.sqrt(mu)

    def y_of(z):
        c, s = _stumpff(z)
        return n1 + n2 + A * (z * s - 1) / np.sqrt(c), c, s

    def f_of(z):
        y, c, s = y_of(z)
        bad = y < 0
        y = np.maximum(y, 0)
        f = (y / c) ** 1.5 * s + A * np.sqrt(y) - sq_mu * tof
        return np.where(bad, -np.inf, f), y

    lo = np.full_like(tof, -60.0)
    hi = np.full_like(tof, 4 * math.pi ** 2 - 1e-6)
    f_hi, _ = f_of(hi)
    ok = f_hi > 0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        f_mid, _ = f_of(mid)
        go_up = f_mid < 0
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    z = (lo + hi) / 2
    y, c, s = y_of(z)
    y = np.maximum(y, 1e-9)
    f = 1 - y / n1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / n2
    g = np.where(np.abs(g) < 1e-9, 1e-9, g)
    v1 = (r2 - f[..., None] * r1) / g[..., None]
    v2 = (gdot[..., None] * r2 - r1) / g[..., None]
    ok = ok & np.isfinite(v1).all(-1) & np.isfinite(v2).all(-1) & (np.abs(A) > 1e-3)
    return v1, v2, ok
