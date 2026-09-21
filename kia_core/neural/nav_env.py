"""МЕЖПЛАНЕТНЫЙ ТРЕНАЖЁР НАВИГАТОРА.

Одна миссия = одно решение. Навигатору показывают обстановку (куда,
зачем, где сейчас планеты, какой у аппарата запас Δv и антенна, какие
настройки связи в мире), он отвечает планом:

    непрерывные оси
      0  фазовый угол вылета         ±180° от гомановского (момент
                                     старта тренажёр находит сам)
      1  поправка времени перелёта   ±20 % к оптимальному (оптимум под
                                     выбранный вылет считает Ламберт)
      2  высота перицентра захвата   от кромки атмосферы до 0.3 SOI
      3  момент коррекции курса      [0.05 … 0.95] пути
    дискретные головы
      0  решение: 0 — лететь, 1 — «Δv не хватит», отказ
      1  сколько ретрансляторов доставить заранее (0…4)
      2  на какую орбиту их ставить: низкая / средняя / высокая
      3  аэрозахват: 0 — нет, 1 — да (торможение атмосферой)

Тренажёр считает миссию как в KSP: положения планет по орбитам из игры,
вылет с НОО 80 км по Ламберту, захват по v∞, посадку по карте Δv,
связь по правилам CommNet (мощность антенн, DSN, заслонение планетой).

Учебная программа (три ступени, как у остальных сетей):
    1. Навигатор    — только орбиты, связь выключена, Δv с запасом
    2. Навигатор+   — посадки и высокие орбиты, атмосферы, разброс
                      двигателя, Δv впритык: надо уметь отказаться
    3. Навигатор++  — CommNet: нет связи → доставить ретрансляторы,
                      столько, сколько нужно, и туда, где они нужны
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .solar import HOME, PLANETS, GAS_GIANTS, SolarSystem, lambert

OBJECTIVES = ("land", "low", "high")
OBJECTIVE_RU = {"land": "посадка", "low": "низкая орбита", "high": "высокая орбита"}
RELAY_CLASSES = ("низкая", "средняя", "высокая")

CONTINUOUS_DIM = 4
OBS_DIM = 6 + 3 + 22                    # планеты + цели + обстановка
DISCRETE_SIZES = (2, 5, 3, 2)
PARK_ALT = 80_000.0
YEAR = 9_203_545.0                      # год Кербина, с
DSN = {1: 2.0e9, 2: 5.0e10, 3: 2.5e11}
RELAY_POWER = 1.0e11                    # RA-100: ретранслятор дальней связи
RELAY_COST = 0.4                        # награды за каждый доставленный спутник

# Покрытие поверхности/низкой орбиты n ретрансляторами на орбите класса k
# (доля времени, когда аппарат видит хоть один). Высокая орбита видит
# планету почти целиком, двух спутников напротив хватает; низкую
# орбиту заслоняет сама планета — нужно четыре.
COVERAGE = np.array([
    # n = 0     1     2     3     4
    [0.00, 0.30, 0.55, 0.80, 0.97],     # низкая
    [0.00, 0.45, 0.80, 0.97, 0.99],     # средняя
    [0.00, 0.50, 0.96, 0.99, 1.00],     # высокая
])
RELAY_ALT = (0.6, 3.0, None)            # доли радиуса; высокая = 0.25 SOI


@dataclass
class StageRules:
    number: int
    objectives: tuple[str, ...]
    commnet: bool
    tight_dv: bool
    dispersion: bool


STAGE_RULES = {
    1: StageRules(1, ("low",), commnet=False, tight_dv=False, dispersion=False),
    2: StageRules(2, OBJECTIVES, commnet=False, tight_dv=True, dispersion=True),
    3: StageRules(3, OBJECTIVES, commnet=True, tight_dv=True, dispersion=True),
}


# ==========================================================================
class NavigatorEnv:
    """Пачка миссий. reset() → наблюдения, step(действия) → награды, итоги."""

    def __init__(self, batch: int = 1024, stage: int = 1, seed: int | None = None,
                 system: SolarSystem | None = None):
        self.S = system or SolarSystem()
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        self.set_stage(stage)
        k = self.S[HOME]
        self.r_park = k.radius + PARK_ALT
        self.v_park = math.sqrt(k.mu / self.r_park)
        self.hohmann = {t: self.S.hohmann(t) for t in PLANETS}
        self.synodic = {t: self.S.synodic(t) for t in PLANETS}
        # Гомановский фазовый угол — опорная точка для оси 0
        self.phase_h = {t: (math.pi - 2 * math.pi * self.hohmann[t][2] / self.S[t].period)
                        % (2 * math.pi) for t in PLANETS}
        self.ctx: dict = {}

    def set_stage(self, stage: int) -> None:
        self.stage = int(stage)
        self.rules = STAGE_RULES[self.stage]

    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    def reset(self) -> np.ndarray:
        n, rng, rules = self.batch, self.rng, self.rules
        ti = rng.integers(0, len(PLANETS), n)
        oi = rng.integers(0, len(rules.objectives), n)
        obj = np.array([OBJECTIVES.index(rules.objectives[i]) for i in oi])
        giant = np.array([PLANETS[t] in GAS_GIANTS for t in ti])
        obj = np.where(giant & (obj == 0), 1 + rng.integers(0, 2, n), obj)
        ut0 = rng.uniform(0, 10 * YEAR, n)

        c = {"target": ti, "objective": obj, "ut0": ut0}
        c["heatshield"] = rng.random(n) < (0.6 if rules.tight_dv else 1.0)
        c["chutes"] = rng.random(n) < (0.7 if rules.tight_dv else 1.0)
        if rules.commnet:
            c["commnet"] = rng.random(n) < 0.85
            c["require"] = c["commnet"] & (rng.random(n) < 0.6)
            c["range_mod"] = rng.uniform(0.5, 1.5, n)
            c["dsn"] = rng.integers(1, 4, n)
            c["antenna"] = 10 ** rng.uniform(np.log10(5e3), np.log10(1e11), n)
            c["relays_there"] = np.where(rng.random(n) < 0.15, rng.integers(1, 5, n), 0)
        else:
            c["commnet"] = np.zeros(n, bool)
            c["require"] = np.zeros(n, bool)
            c["range_mod"] = np.ones(n)
            c["dsn"] = np.full(n, 3)
            c["antenna"] = np.full(n, 1e11)
            c["relays_there"] = np.zeros(n, int)
        self.ctx = c
        # Эталон нужен и для запаса Δv, и для оценки: считается один раз
        self.expert = self.expert_plan()
        need = self.expert["total"]
        if rules.tight_dv:
            c["dv"] = need * rng.uniform(0.8, 1.6, n)
        else:
            c["dv"] = need * rng.uniform(1.3, 1.8, n)
        self.expert["abort"] = (c["dv"] < need) | self.expert["comm_impossible"]
        return self.observe()

    # ------------------------------------------------------------------
    def observe(self) -> np.ndarray:
        c = self.ctx
        n = self.batch
        feats = np.zeros((n, self.obs_dim), dtype=np.float32)
        feats[np.arange(n), c["target"]] = 1.0
        feats[np.arange(n), len(PLANETS) + c["objective"]] = 1.0
        j = len(PLANETS) + len(OBJECTIVES)
        t_names = [PLANETS[t] for t in c["target"]]
        phase = np.zeros(n)
        for name in set(t_names):
            m = np.array([tn == name for tn in t_names])
            phase[m] = self.S.phase_angle(name, c["ut0"][m])
        syn = np.array([self.synodic[t] for t in t_names])
        hoh = np.array([self.hohmann[t] for t in t_names])        # ej, v∞, tof
        bodies = [self.S[t] for t in t_names]
        radius = np.array([b.radius for b in bodies])
        atm = np.array([b.atmosphere for b in bodies])
        soi = np.array([b.soi for b in bodies])
        g0 = np.array([b.g0 for b in bodies])
        sma = np.array([b.sma for b in bodies])
        dmax = sma + self.S[HOME].sma
        cols = [
            np.sin(phase), np.cos(phase),
            np.log10(syn / 86400), np.log10(hoh[:, 2] / 86400),
            hoh[:, 0] / 1000, hoh[:, 1] / 1000,
            radius / 1e6, atm / 1e5, np.log10(soi), g0 / 10,
            # Точная цена миссии от планировщика (Ламберт по окну, захват,
            # посадка) делённая на запас аппарата. Лететь или отказаться —
            # это сравнение с единицей, а не догадка: без этого входа сеть
            # ошибалась в 8 % решений «лететь/отказ» и застряла на 87 %.
            c["dv"] / 1000, np.clip(self.expert["total"] / c["dv"], 0.0, 3.0),
            c["heatshield"].astype(float), c["chutes"].astype(float),
            np.log10(c["antenna"]) / 10, c["commnet"].astype(float),
            c["require"].astype(float), c["range_mod"], c["dsn"] / 3,
            c["relays_there"] / 4, np.log10(dmax) / 12,
            # Окно вылета, рассчитанное планировщиком (фазовая ось). Лучший
            # угол зависит от того, где на орбитах окажутся планеты, — это
            # точная математика; сеть решает, что с этим делать, а не
            # пересчитывает Ламберта в уме (без входа — ошибка ~24°).
            self.phase_axis(t_names, self.expert["phi"]),
        ]
        feats[:, j:j + len(cols)] = np.stack(cols, axis=-1)
        return feats

    def expert_hint(self) -> np.ndarray:
        """Грубая оценка полной Δv по Гоману — то, что уже знает математик."""
        table = np.zeros(len(PLANETS))
        for i, t in enumerate(PLANETS):
            ej, vinf, _ = self.hohmann[t]
            b = self.S[t]
            rp = b.radius + b.low_orbit()
            table[i] = ej + math.sqrt(vinf ** 2 + 2 * b.mu / rp) - math.sqrt(b.mu / rp)
        return table[self.ctx["target"]]

    # ------------------------------------------------------------------
    def time_to_phase(self, names, ut0, phi):
        """Через сколько секунд фазовый угол цели станет равен phi.

        Как делает игрок: смотрит, когда планета придёт в нужное
        положение относительно Кербина. Первое совпадение в пределах
        синодического периода."""
        n = len(names)
        out = np.zeros(n)
        for name in set(names):
            m = np.array([t == name for t in names])
            syn = self.synodic[name]
            steps = 96
            ts = ut0[m][:, None] + np.linspace(0, syn, steps + 1)[None, :]
            ph = self.S.phase_angle(name, ts)
            d = np.angle(np.exp(1j * (ph - phi[m][:, None])))
            cross = (np.sign(d[:, :-1]) != np.sign(d[:, 1:])) & (np.abs(d[:, :-1] - d[:, 1:]) < math.pi)
            first = np.where(cross.any(1), cross.argmax(1), np.abs(d).argmin(1))
            rows = np.arange(m.sum())
            d0 = d[rows, first]
            d1 = d[rows, np.minimum(first + 1, steps)]
            den = d0 - d1
            frac = np.divide(d0, den, out=np.zeros_like(d0), where=np.abs(den) > 1e-12)
            frac = np.clip(np.where(cross.any(1), frac, 0.0), 0, 1)
            out[m] = (first + frac) * syn / steps
        return out

    def best_tof(self, names, ut_dep):
        """Лучшее время перелёта (доля гомановского) для заданного вылета.

        Точная математика — считает Ламберт, а не сеть: у времени перелёта
        два близких по цене решения (короткая и длинная траектория), и
        учиться «угадывать», какое из них выбрал учитель, бессмысленно."""
        n = len(names)
        hoh_tof = np.array([self.hohmann[t][2] for t in names])
        best = np.full(n, np.inf)
        best_f = np.ones(n)
        for grid in (np.linspace(0.55, 1.45, 10), None):
            if grid is None:
                base = best_f.copy()
                grid_rows = [base + d for d in np.linspace(-0.05, 0.05, 5)]
            else:
                grid_rows = [np.full(n, g) for g in grid]
            for tf in grid_rows:
                tf = np.clip(tf, 0.5, 1.5)
                ej, vinf = self._transfer(names, ut_dep, tf * hoh_tof)
                cost = ej + vinf * 0.5
                better = cost < best
                best = np.where(better, cost, best)
                best_f = np.where(better, tf, best_f)
        return best_f

    def phase_axis(self, names, phi):
        """Фазовый угол → значение оси 0 в [-1, 1]."""
        ph = np.array([self.phase_h[t] for t in names])
        return np.angle(np.exp(1j * (phi - ph))) / math.pi

    # ------------------------------------------------------------------
    # Физика миссии
    # ------------------------------------------------------------------
    def _transfer(self, targets, ut_dep, tof):
        """Ламберт пачкой: Δv вылета с НОО и v∞ прилёта."""
        ej = np.full(len(targets), np.inf)
        vinf_arr = np.full(len(targets), np.inf)
        r1, v1 = self.S.state(HOME, ut_dep)
        k = self.S[HOME]
        for name in set(targets):
            m = np.array([t == name for t in targets])
            r2, v2 = self.S.state(name, ut_dep[m] + tof[m])
            va, vb, ok = lambert(r1[m], r2, tof[m], self.S.sun.mu)
            vi = np.linalg.norm(va - v1[m], axis=-1)
            e = np.sqrt(vi ** 2 + 2 * k.mu / self.r_park) - self.v_park
            va_ = np.linalg.norm(vb - v2, axis=-1)
            ej[m] = np.where(ok, e, np.inf)
            vinf_arr[m] = np.where(ok, va_, np.inf)
        return ej, vinf_arr

    def _arrival(self, targets, objective, vinf, cap_alt, aero, heat, chutes):
        """Δv у цели: захват, выход на рабочую орбиту, посадка.

        Возвращает (Δv, провал аэрозахвата)."""
        n = len(targets)
        dv = np.zeros(n)
        burned = np.zeros(n, bool)
        for name in set(targets):
            m = np.array([t == name for t in targets])
            b = self.S[name]
            low, high = b.low_orbit(), b.high_orbit()
            goal = np.where(objective[m] == 2, high, low)
            rp = b.radius + cap_alt[m]
            v_hyp = np.sqrt(vinf[m] ** 2 + 2 * b.mu / rp)
            # Пропульсивный захват сразу на круговую на высоте перицентра
            capture = v_hyp - np.sqrt(b.mu / rp)
            use_aero = aero[m] & (b.atmosphere > 0)
            if b.atmosphere > 0:
                depth = cap_alt[m] / b.atmosphere          # доля толщи атмосферы
                in_corridor = (depth > 0.35) & (depth < 0.7)
                too_deep = depth <= 0.35
                shield_ok = heat[m] | (vinf[m] < 2_500.0)
                ok = in_corridor & shield_ok
                burned[m] = use_aero & (~ok | too_deep)
                # Атмосфера гасит скорость; ещё несколько проходов через
                # верхние слои снижают апоцентр бесплатно. Остаётся поднять
                # перицентр из атмосферы и скруглить на рабочей высоте.
                apo = b.radius + np.minimum(b.soi * 0.1,
                                            np.maximum(goal * 3.0, b.atmosphere * 2.0))
                r_goal = b.radius + goal
                raise_pe = 60.0 + abs(np.sqrt(b.mu * (2 / apo - 2 / (apo + b.radius + b.atmosphere)))
                                      - np.sqrt(b.mu * (2 / apo - 2 / (apo + r_goal))))
                circ = np.abs(np.sqrt(b.mu / r_goal)
                              - np.sqrt(b.mu * (2 / r_goal - 1 / ((apo + r_goal) / 2))))
                aero_dv = raise_pe + circ
                capture = np.where(use_aero, aero_dv, capture)
                start = np.where(use_aero, goal, cap_alt[m])
            else:
                burned[m] = use_aero                        # атмосферы нет — «аэрозахват» = промах
                start = cap_alt[m]
            # Переход с орбиты захвата на рабочую (Гоман, два импульса)
            r1 = b.radius + start
            r2 = b.radius + goal
            a = (r1 + r2) / 2
            move = (np.abs(np.sqrt(b.mu * (2 / r1 - 1 / a)) - np.sqrt(b.mu / r1))
                    + np.abs(np.sqrt(b.mu / r2) - np.sqrt(b.mu * (2 / r2 - 1 / a))))
            total = capture + move
            # Посадка с низкой орбиты
            land = objective[m] == 0
            v_low = math.sqrt(b.mu / (b.radius + low))
            if b.atmosphere > 0:
                landing = np.where(chutes[m], 0.12 * v_low + 120.0, 0.55 * v_low + 200.0)
            else:
                landing = np.full(m.sum(), v_low * 1.12 + 60.0)
            total = total + np.where(land, landing, 0.0)
            dv[m] = total
        return dv, burned

    def _comm(self, targets, objective, relays, relay_class):
        """Связь с домом: (надёжность 0…1, нужна ли связь вообще, невозможна ли).

        Правило CommNet: дальность = √(мощность₁·мощность₂), мощность
        аппарата умножается на множитель дальности, DSN — на свой."""
        c = self.ctx
        n = len(targets)
        dsn = np.array([DSN[int(d)] for d in c["dsn"]])
        dmax = np.array([self.S[t].sma + self.S[HOME].sma for t in targets])
        direct_range = np.sqrt(c["antenna"] * c["range_mod"] * dsn)
        relay_range = np.sqrt(RELAY_POWER * c["range_mod"] * dsn)
        direct = direct_range >= dmax
        relay_reach = relay_range >= dmax
        present = relays + c["relays_there"]
        present = np.minimum(present, 4)
        cls = np.where(relays > 0, relay_class, 2)          # уже стоящие — высокие
        cover = COVERAGE[cls, present]
        cover = np.where(relay_reach, cover, 0.0)
        # Без ретрансляторов аппарат на низкой орбите/поверхности видит дом
        # примерно половину времени, на высокой — почти всегда
        self_vis = np.where(objective == 2, 0.95, 0.5)
        reliability = np.where(direct, np.maximum(self_vis, cover), cover)
        impossible = c["require"] & ~direct & ~relay_reach
        return reliability, impossible

    # ------------------------------------------------------------------
    def expert_plan(self) -> dict:
        """Точный планировщик — «знания», которыми сеть учится вначале.

        Перебор окна вылета и времени перелёта (сетка × Ламберт), захват на
        рабочую высоту или аэрозахват в середине коридора, коррекция
        пораньше, минимальное число ретрансляторов, которое даёт
        надёжную связь."""
        c = self.ctx
        n = self.batch
        names = [PLANETS[t] for t in c["target"]]
        W, F = 20, 8
        best_total = np.full(n, np.inf)
        best_wait = np.zeros(n)
        best_tof = np.ones(n)
        wait_f = np.linspace(0, 1, W, endpoint=False)
        tof_f = np.linspace(0.6, 1.4, F)
        syn = np.array([self.synodic[t] for t in names])
        hoh_tof = np.array([self.hohmann[t][2] for t in names])
        for wf in wait_f:
            for tf in tof_f:
                ej, vinf = self._transfer(names, c["ut0"] + wf * syn, tf * hoh_tof)
                cost = ej + vinf * 0.5              # грубая цена прилёта для выбора окна
                better = cost < best_total
                best_total = np.where(better, cost, best_total)
                best_wait = np.where(better, wf, best_wait)
                best_tof = np.where(better, tf, best_tof)
        # Уточнение окна мелкой сеткой вокруг найденного
        step_w, step_t = 1.0 / W, (tof_f[1] - tof_f[0])
        base_w, base_t = best_wait.copy(), best_tof.copy()
        for dw in np.linspace(-step_w, step_w, 9):
            for dt in np.linspace(-step_t, step_t, 5):
                wf = np.clip(base_w + dw, 0, 1)
                tf = np.clip(base_t + dt, 0.5, 1.5)
                ej, vinf = self._transfer(names, c["ut0"] + wf * syn, tf * hoh_tof)
                cost = ej + vinf * 0.5
                better = cost < best_total
                best_total = np.where(better, cost, best_total)
                best_wait = np.where(better, wf, best_wait)
                best_tof = np.where(better, tf, best_tof)
        ej, vinf = self._transfer(names, c["ut0"] + best_wait * syn, best_tof * hoh_tof)
        phi_dep = np.zeros(n)
        for name in set(names):
            m = np.array([t == name for t in names])
            phi_dep[m] = self.S.phase_angle(name, c["ut0"][m] + best_wait[m] * syn[m])
        # Итог эталона — тем же путём, что и план сети: угол → момент
        # вылета → оптимальный перелёт. Иначе запас Δv и оценка считались
        # бы для траектории, которую сеть повторить не может.
        best_wait = self.time_to_phase(names, c["ut0"], phi_dep) / syn
        best_tof = self.best_tof(names, c["ut0"] + best_wait * syn)
        ej, vinf = self._transfer(names, c["ut0"] + best_wait * syn, best_tof * hoh_tof)
        # Высота захвата: рабочая орбита, а с атмосферой и щитом — аэрозахват
        low = np.array([self.S[t].low_orbit() for t in names])
        high = np.array([self.S[t].high_orbit() for t in names])
        atm = np.array([self.S[t].atmosphere for t in names])
        obj = c["objective"]
        goal = np.where(obj == 2, high, low)
        aero = (atm > 0) & (c["heatshield"] | (vinf < 2_500.0)) & (obj != 2)
        cap_alt = np.where(aero, atm * 0.5, goal)
        arr, _ = self._arrival(names, obj, vinf, cap_alt, aero, c["heatshield"], c["chutes"])
        corr = 0.004 * ej * (1 + 2 * 0.1) + 5.0 if self.rules.dispersion else np.zeros(n)
        total = ej + arr + corr
        # Связь: минимальное число спутников, дающее ≥ 0.95 надёжности
        relays = np.zeros(n, int)
        rclass = np.full(n, 2)
        need_comm = c["require"] | c["commnet"]
        found = np.zeros(n, bool)
        rel0, impossible = self._comm(names, obj, np.zeros(n, int), rclass)
        found |= (rel0 >= 0.95) | ~need_comm
        for count in range(1, 5):
            for cls in (2, 1, 0):
                r, _ = self._comm(names, obj, np.full(n, count), np.full(n, cls))
                hit = ~found & (r >= 0.95)
                relays = np.where(hit, count, relays)
                rclass = np.where(hit, cls, rclass)
                found |= hit
        return {"wait": best_wait, "phi": phi_dep, "tof": best_tof, "cap_alt": cap_alt, "aero": aero,
                "corr_t": np.full(n, 0.1), "relays": relays, "relay_class": rclass,
                "total": total, "ejection": ej, "vinf": vinf,
                "comm_impossible": impossible}

    # ------------------------------------------------------------------
    def decode(self, continuous: np.ndarray, discrete: np.ndarray) -> dict:
        c = self.ctx
        names = [PLANETS[t] for t in c["target"]]
        u = (np.clip(continuous, -1, 1) + 1) / 2
        atm = np.array([self.S[t].atmosphere for t in names])
        soi = np.array([self.S[t].soi for t in names])
        floor = np.maximum(atm * 0.2, 5_000.0)
        top = soi * 0.3
        cap_alt = floor * (top / floor) ** u[:, 2]          # логарифмическая шкала
        ph = np.array([self.phase_h[t] for t in names])
        phi = np.mod(ph + np.clip(continuous[:, 0], -1, 1) * math.pi, 2 * math.pi)
        syn = np.array([self.synodic[t] for t in names])
        wait = self.time_to_phase(names, c["ut0"], phi) / syn
        tof = self.best_tof(names, c["ut0"] + wait * syn) * (1 + 0.2 * np.clip(continuous[:, 1], -1, 1))
        return {"wait": wait, "phi": phi, "tof": tof, "cap_alt": cap_alt,
                "corr_t": 0.05 + 0.9 * u[:, 3], "abort": discrete[:, 0] == 1,
                "relays": discrete[:, 1], "relay_class": discrete[:, 2],
                "aero": discrete[:, 3] == 1}

    def encode_expert(self) -> tuple[np.ndarray, np.ndarray]:
        """План эталона в координатах действий сети — для предобучения."""
        e, c = self.expert, self.ctx
        names = [PLANETS[t] for t in c["target"]]
        atm = np.array([self.S[t].atmosphere for t in names])
        soi = np.array([self.S[t].soi for t in names])
        floor = np.maximum(atm * 0.2, 5_000.0)
        top = soi * 0.3
        u2 = np.log(np.clip(e["cap_alt"], floor, top) / floor) / np.log(top / floor)
        u = np.stack([np.zeros(len(names)), np.full(len(names), 0.5), u2,
                      (e["corr_t"] - 0.05) / 0.9], axis=-1)
        cont = np.clip(u * 2 - 1, -0.98, 0.98).astype(np.float32)
        cont[:, 0] = np.clip(self.phase_axis(names, e["phi"]), -0.98, 0.98)
        disc = np.stack([e["abort"].astype(int), e["relays"], e["relay_class"],
                         e["aero"].astype(int)], axis=-1)
        return cont, disc

    # ------------------------------------------------------------------
    def step(self, continuous: np.ndarray, discrete: np.ndarray):
        """Проигрывает миссии по плану. Возвращает (награды, итоги)."""
        a = self.decode(continuous, discrete)
        c, e, rng = self.ctx, self.expert, self.rng
        n = self.batch
        names = [PLANETS[t] for t in c["target"]]
        syn = np.array([self.synodic[t] for t in names])
        hoh_tof = np.array([self.hohmann[t][2] for t in names])
        ej, vinf = self._transfer(names, c["ut0"] + a["wait"] * syn, a["tof"] * hoh_tof)
        arr, burned = self._arrival(names, c["objective"], vinf, a["cap_alt"], a["aero"],
                                    c["heatshield"], c["chutes"])
        if self.rules.dispersion:
            err = 0.004 * ej * np.abs(rng.normal(1.0, 0.3, n))
            corr = err * (1 + 2 * a["corr_t"]) + 5.0
        else:
            corr = np.zeros(n)
        total = ej + arr + corr
        feasible = np.isfinite(total) & (total <= c["dv"])

        reliability, impossible = self._comm(names, c["objective"], a["relays"], a["relay_class"])
        # Без связи при «управление только со связью» аппарат теряется в
        # момент, когда он не на прямой видимости — розыгрыш по надёжности
        comm_lost = c["require"] & (rng.random(n) > reliability)
        science_lost = c["commnet"] & ~c["require"] & (reliability < 0.9)

        success = ~a["abort"] & feasible & ~burned & ~comm_lost
        reward = np.zeros(n)
        expert_total = np.maximum(e["total"], 1.0)
        eff = np.clip(expert_total / np.where(np.isfinite(total), total, 1e12), 0, 1.2)
        days = (a["wait"] * syn + a["tof"] * hoh_tof) / 21_600.0
        reward = np.where(success, 6.0 + 4.0 * eff - 0.0005 * days, reward)
        reward = np.where(~a["abort"] & ~feasible, -5.0, reward)
        reward = np.where(~a["abort"] & burned, -6.0, reward)
        reward = np.where(~a["abort"] & feasible & ~burned & comm_lost, -5.0, reward)
        reward = np.where(success & science_lost, reward - 1.0, reward)
        # Отказ: верный — награда, лишний — штраф
        right_abort = e["abort"]
        reward = np.where(a["abort"], np.where(right_abort, 4.0, -3.0), reward)
        # Спутники стоят денег и времени: каждый лишний — минус
        needed = e["relays"]
        reward = reward - RELAY_COST * a["relays"] - 0.05 * a["relays"] * a["relay_class"]
        reward = np.where(~a["abort"] & c["require"] & (a["relays"] < needed) & success,
                          reward - 1.0, reward)
        info = {"success": success, "abort": a["abort"], "right_abort": right_abort,
                "burned": burned, "comm_lost": comm_lost, "feasible": feasible,
                "target": c["target"], "objective": c["objective"], "total": total,
                "expert_total": e["total"], "relays": a["relays"], "relays_needed": needed,
                "ok": np.where(right_abort, a["abort"], success)}
        return reward.astype(np.float32), info
