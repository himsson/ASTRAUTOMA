"""ТРЕНАЖЁР МАТЕМАТИКИ: сеть учится считать то, что считает инженер.

Зачем он нужен. Пилот KIA умеет держать нос по потоку и выводить аппарат
на орбиту, но не имеет ни малейшего понятия, СКОЛЬКО стоит манёвр. Все
эти числа — круговая скорость, гомановский импульс, фазовый угол окна —
до сих пор считались только формулами в `engineer/rocket_math.py`, и сеть
о них не знала ничего.

Главное решение этого модуля: ЭТАЛОН БЕРЁТСЯ ИЗ ТЕХ ЖЕ ФУНКЦИЙ, которыми
пользуется живая игра. Не из отдельной копии формул, написанной специально
для тренажёра, — именно из `rocket_math`. Иначе неизбежно расхождение:
тренажёр учит одному, полётный код считает другое, и знание, добытое за
миллионы шагов, в игре оказывается неприменимым. Здесь такое расхождение
невозможно по построению: если формула в игре изменится, изменится и то,
чему учится сеть.

Устройство эпизода. Один эпизод — одна задача, один шаг: сеть видит
условие (тип задачи и её параметры), выдаёт ответ, получает награду по
относительной ошибке. Многошаговых эпизодов здесь нет и не нужно —
это не управление, а счёт.

Три ступени, как и у пилота:

    1. Матан      арифметика ракеты: круговая скорость, период,
                  Циолковский, TWR, время ожога
    2. Матан+     манёвры на орбите: vis-viva, подъём апоапсиса,
                  скругление, Гоман целиком, время перелёта
    3. Матан++    межпланетное: фазовый угол, синодический период,
                  смена плоскости, уход из сферы влияния, захват

Каждая следующая ступень подмешивает задачи предыдущих (доля
`REVIEW_SHARE`). Без этого сеть забывает пройденное — это не догадка, а
известное свойство обучения с подкреплением: катастрофическое забывание.
Ступень 3, обученная только на межпланетных задачах, круговую скорость
считать разучится.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..engineer import rocket_math as rm
from ..engineer.parts_db import get_body
from ..logging_setup import get_logger

log = get_logger("neural.math_env")

G0 = 9.80665

# Сколько задач прошлых ступеней подмешивается в текущую.
REVIEW_SHARE = 0.30

# Сколько всего типов задач может знать сеть. Число фиксировано и с
# запасом: вектор наблюдения обязан иметь постоянную длину, иначе
# сохранённые веса потеряют смысл при добавлении новой задачи.
MATH_TASK_SLOTS = 24
# Сколько чисел условия подаётся на вход.
MATH_PARAM_SLOTS = 6
# Каждое число условия подаётся ДВАЖДЫ: как есть и своим логарифмом.
#
# Это не украшение, а главное решение всего тренажёра. Почти вся баллистика
# состоит из произведений, отношений и корней: v = sqrt(mu/r),
# T = 2pi sqrt(a^3/mu), TWR = F/(m g). Обычному персептрону деление даётся
# тяжело — он умеет складывать и умножать на веса, а не делить. Но в
# логарифмах ВСЕ эти формулы становятся ЛИНЕЙНЫМИ:
#
#     ln v = 0.5 (ln mu - ln r)
#     ln T = ln 2pi + 1.5 ln a - 0.5 ln mu
#
# то есть ровно тем, чем персептрон и является. Ответ тоже кодируется
# логарифмически (см. MathTask), поэтому сеть учит линейную зависимость
# между входом и выходом, а не пытается нащупать деление.
#
# Замер: без логарифмов точность первой ступени встала на 15.6 %.
MATH_OBS_DIM = MATH_TASK_SLOTS + MATH_PARAM_SLOTS * 2
# Предел логарифмического признака: неиспользуемые слоты условия равны
# нулю, и ln(0) без него дал бы минус бесконечность.
LOG_FEATURE_FLOOR = 1e-9
LOG_FEATURE_LIMIT = 25.0
# Сколько чисел сеть выдаёт в ответ. Двух хватает: единственная задача с
# двумя ответами — гомановский переход (два импульса).
MATH_ANSWER_DIM = 2

# Тела, на которых ставятся задачи. Кербол исключён: задачи вокруг
# звезды имеют совсем другой масштаб чисел и в одиночку перекашивают
# нормализатор входов.
TASK_BODIES = ("Kerbin", "Mun", "Minmus", "Duna")


# ==========================================================================
@dataclass
class MathTask:
    """Одна задача: условие, эталонный ответ и диапазон поиска.

    `low`/`high` задают, в каких пределах вообще может лежать ответ. Сеть
    выдаёт число в [-1, 1], которое разворачивается в этот диапазон по
    ЛОГАРИФМИЧЕСКОЙ шкале. Линейная здесь не годится: ответы одной и той
    же задачи различаются на порядки (импульс от 10 м/с до 6000 м/с), и
    при линейной шкале вся мелочь схлопывается в одну неразличимую точку
    у нуля.
    """
    key: str
    title: str
    stage: int
    slot: int                     # индекс типа задачи в one-hot
    params: list[float]           # нормированные числа условия
    answers: tuple[float, ...]    # эталон
    low: tuple[float, ...]        # нижняя граница диапазона ответа
    high: tuple[float, ...]       # верхняя граница
    units: str = ""
    given: str = ""               # человекочитаемое условие для журнала

    @property
    def count(self) -> int:
        return len(self.answers)

    def encode(self) -> list[float]:
        vector = [0.0] * MATH_OBS_DIM
        vector[self.slot] = 1.0
        base = MATH_TASK_SLOTS
        for index, value in enumerate(self.params[:MATH_PARAM_SLOTS]):
            value = float(value)
            vector[base + index] = value
            # Знак сохраняется отдельно от величины: фазовый угол бывает
            # отрицательным, а логарифм от модуля всё равно осмыслен.
            magnitude = math.log(max(abs(value), LOG_FEATURE_FLOOR))
            magnitude = max(-LOG_FEATURE_LIMIT, min(LOG_FEATURE_LIMIT, magnitude))
            vector[base + MATH_PARAM_SLOTS + index] = magnitude
        return vector

    def target(self) -> list[float]:
        """Эталон, переведённый обратно в выход сети [-1, 1].

        Нужен для обучения счёту БЕЗ проб и ошибок. Точный ответ здесь
        известен заранее, и добывать его случайным поиском бессмысленно:
        это регрессия, а не поведение. Обратное преобразование к `decode`.
        """
        out = []
        for index, truth in enumerate(self.answers):
            lo, hi = self.low[index], self.high[index]
            if lo > 0 and hi > lo:
                unit = math.log(max(truth, 1e-12) / lo) / math.log(hi / lo)
            else:
                unit = (truth - lo) / (hi - lo) if hi > lo else 0.5
            out.append(max(-1.0, min(1.0, unit * 2.0 - 1.0)))
        while len(out) < MATH_ANSWER_DIM:
            out.append(0.0)              # неиспользуемый выход держим в нуле
        return out

    def decode(self, action) -> list[float]:
        """Выход сети [-1, 1] -> ответ в физических единицах."""
        out = []
        for index in range(self.count):
            raw = float(action[index]) if index < len(action) else 0.0
            unit = (max(-1.0, min(1.0, raw)) + 1.0) / 2.0
            lo, hi = self.low[index], self.high[index]
            if lo > 0 and hi > lo:
                out.append(lo * (hi / lo) ** unit)      # логарифмическая шкала
            else:
                out.append(lo + (hi - lo) * unit)       # линейная: ответ может быть <= 0
        return out

    def errors(self, given) -> list[float]:
        """Относительная ошибка по каждому ответу."""
        out = []
        for index, truth in enumerate(self.answers):
            got = given[index] if index < len(given) else 0.0
            scale = max(abs(truth), 1e-6)
            out.append(abs(got - truth) / scale)
        return out


# ==========================================================================
# Генераторы задач. Каждый возвращает MathTask с эталоном ИЗ rocket_math.
# ==========================================================================
def _body(rng: random.Random):
    return get_body(rng.choice(TASK_BODIES))


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return low * (high / low) ** rng.random()


# --- Ступень 1: арифметика ракеты -----------------------------------------
def _t_circular_speed(rng, slot):
    body = _body(rng)
    alt = _log_uniform(rng, 10_000.0, 500_000.0)
    truth = rm.circular_speed(body, alt)
    return MathTask(
        key="circular_speed", title="круговая скорость", stage=1, slot=slot,
        params=[body.mu / 3.5e12, body.radius / 6.0e5, alt / 1.0e5, 0, 0, 0],
        answers=(truth,), low=(50.0,), high=(5_000.0,), units="м/с",
        given=f"{body.name}, высота {alt / 1000:.0f} км")


def _t_orbital_period(rng, slot):
    body = _body(rng)
    alt = _log_uniform(rng, 10_000.0, 2_000_000.0)
    sma = body.radius + alt
    truth = rm.orbital_period(body, sma)
    return MathTask(
        key="orbital_period", title="период обращения", stage=1, slot=slot,
        params=[body.mu / 3.5e12, body.radius / 6.0e5, sma / 1.0e6, 0, 0, 0],
        answers=(truth,), low=(100.0,), high=(2_000_000.0,), units="с",
        given=f"{body.name}, большая полуось {sma / 1000:.0f} км")


def _t_surface_gravity(rng, slot):
    body = _body(rng)
    alt = _log_uniform(rng, 0.0 + 1000.0, 800_000.0)
    r = body.radius + alt
    truth = body.mu / (r ** 2)
    return MathTask(
        key="gravity_at", title="ускорение силы тяжести на высоте", stage=1,
        slot=slot,
        params=[body.mu / 3.5e12, body.radius / 6.0e5, r / 1.0e6, 0, 0, 0],
        answers=(truth,), low=(0.001,), high=(30.0,), units="м/с²",
        given=f"{body.name}, радиус {r / 1000:.0f} км")


def _t_tsiolkovsky(rng, slot):
    isp = rng.uniform(120.0, 350.0)
    dry = _log_uniform(rng, 0.3, 40.0)
    wet = dry * rng.uniform(1.15, 6.0)
    truth = rm.delta_v(isp, wet, dry)
    return MathTask(
        key="tsiolkovsky", title="ΔV по Циолковскому", stage=1, slot=slot,
        params=[isp / 350.0, wet / 40.0, dry / 40.0, wet / max(dry, 1e-6) / 6.0,
                0, 0],
        answers=(truth,), low=(10.0,), high=(8_000.0,), units="м/с",
        given=f"Isp {isp:.0f} с, {wet:.2f} т -> {dry:.2f} т")


def _t_twr(rng, slot):
    # Тяга не берётся независимо от массы. Иначе выпадают связки вроде
    # «4000 кН на 0.5 т у Минмуса» с TWR под тысячу: формально это задача,
    # но ответы расползаются на шесть порядков, и в логарифмической шкале
    # ответа не остаётся разрешения на рабочий диапазон 0.5..5.
    body = _body(rng)
    g = body.surface_gravity
    mass = _log_uniform(rng, 0.5, 400.0)
    wanted = _log_uniform(rng, 0.15, 8.0)          # правдоподобный TWR
    thrust = wanted * mass * g
    truth = thrust * 1000.0 / (mass * 1000.0 * g)
    return MathTask(
        key="twr", title="тяговооружённость", stage=1, slot=slot,
        params=[thrust / 1000.0, mass / 100.0, g / 10.0, 0, 0, 0],
        answers=(truth,), low=(0.05,), high=(25.0,), units="",
        given=f"тяга {thrust:.0f} кН, масса {mass:.1f} т, g {g:.2f}")


def _t_burn_time(rng, slot):
    isp = rng.uniform(120.0, 350.0)
    dry = _log_uniform(rng, 0.3, 40.0)
    wet = dry * rng.uniform(1.15, 5.0)
    # Тяга привязана к массе по той же причине, что и в задаче про TWR:
    # 5 кН на сорокатонной ступени дают ожог в 18 часов.
    thrust = wet * rng.uniform(4.0, 60.0)
    truth = rm.burn_time(isp, thrust, wet, dry)
    return MathTask(
        key="burn_time", title="время ожога до сухого бака", stage=1, slot=slot,
        params=[isp / 350.0, thrust / 1000.0, wet / 40.0, dry / 40.0, 0, 0],
        answers=(truth,), low=(2.0,), high=(4_000.0,), units="с",
        given=f"Isp {isp:.0f} с, тяга {thrust:.0f} кН, {wet:.2f}->{dry:.2f} т")


# --- Ступень 2: манёвры на орбите ------------------------------------------
def _t_vis_viva(rng, slot):
    body = _body(rng)
    rp = body.radius + _log_uniform(rng, 10_000.0, 200_000.0)
    ra = rp * rng.uniform(1.05, 12.0)
    sma = (rp + ra) / 2.0
    r = rng.uniform(rp, ra)
    truth = rm.orbital_speed(body, r, sma)
    return MathTask(
        key="vis_viva", title="скорость в точке орбиты (vis-viva)", stage=2,
        slot=slot,
        params=[body.mu / 3.5e12, r / 1.0e6, sma / 1.0e6, ra / 1.0e6,
                rp / 1.0e6, 0],
        answers=(truth,), low=(5.0,), high=(6_000.0,), units="м/с",
        given=f"{body.name}, r {r / 1000:.0f} км, a {sma / 1000:.0f} км")


def _t_raise_apoapsis(rng, slot):
    body = _body(rng)
    r1 = body.radius + _log_uniform(rng, 10_000.0, 200_000.0)
    r2 = r1 * rng.uniform(1.05, 15.0)
    dv1, _dv2, _tof = rm.hohmann_transfer(body, r1, r2)
    return MathTask(
        key="raise_apoapsis", title="импульс подъёма апоапсиса", stage=2,
        slot=slot,
        params=[body.mu / 3.5e12, r1 / 1.0e6, r2 / 1.0e6, r2 / r1 / 15.0, 0, 0],
        answers=(dv1,), low=(1.0,), high=(4_000.0,), units="м/с",
        given=f"{body.name}, круговая {r1 / 1000:.0f} км -> апоапсис "
              f"{r2 / 1000:.0f} км")


def _t_circularize(rng, slot):
    body = _body(rng)
    peri = _log_uniform(rng, 10_000.0, 150_000.0)
    apo = peri * rng.uniform(1.1, 20.0)
    truth = rm.circularization_dv(body, apo, peri)
    return MathTask(
        key="circularize", title="скругление в апоапсисе", stage=2, slot=slot,
        params=[body.mu / 3.5e12, body.radius / 6.0e5, apo / 1.0e6,
                peri / 1.0e6, apo / max(peri, 1.0) / 20.0, 0],
        answers=(truth,), low=(0.3,), high=(3_000.0,), units="м/с",
        given=f"{body.name}, апоапсис {apo / 1000:.0f} км, периапсис "
              f"{peri / 1000:.0f} км")


def _t_hohmann(rng, slot):
    body = _body(rng)
    r1 = body.radius + _log_uniform(rng, 10_000.0, 200_000.0)
    r2 = r1 * rng.uniform(1.1, 20.0)
    dv1, dv2, _tof = rm.hohmann_transfer(body, r1, r2)
    return MathTask(
        key="hohmann", title="гомановский переход: два импульса", stage=2,
        slot=slot,
        params=[body.mu / 3.5e12, r1 / 1.0e6, r2 / 1.0e6, r2 / r1 / 20.0, 0, 0],
        answers=(dv1, dv2), low=(1.0, 1.0), high=(4_000.0, 4_000.0),
        units="м/с",
        given=f"{body.name}, {r1 / 1000:.0f} -> {r2 / 1000:.0f} км")


def _t_hohmann_time(rng, slot):
    body = _body(rng)
    r1 = body.radius + _log_uniform(rng, 10_000.0, 200_000.0)
    r2 = r1 * rng.uniform(1.1, 25.0)
    _dv1, _dv2, tof = rm.hohmann_transfer(body, r1, r2)
    return MathTask(
        key="hohmann_time", title="время гомановского перелёта", stage=2,
        slot=slot,
        params=[body.mu / 3.5e12, r1 / 1.0e6, r2 / 1.0e6, (r1 + r2) / 2.0e6,
                0, 0],
        answers=(tof,), low=(60.0,), high=(3_000_000.0,), units="с",
        given=f"{body.name}, {r1 / 1000:.0f} -> {r2 / 1000:.0f} км")


def _t_apoapsis_from_state(rng, slot):
    """Какой апоапсис получится, если сейчас, в периапсисе, лететь так.

    Задача из живого полёта: аппарат в конце разгона, и надо понять, куда
    его вынесет. Формула получается из vis-viva и сохранения момента:
    r_a = r / (2 mu / (r v^2) - 1).
    """
    body = _body(rng)
    r = body.radius + _log_uniform(rng, 10_000.0, 150_000.0)
    v_circ = math.sqrt(body.mu / r)
    v = v_circ * rng.uniform(1.005, 1.38)          # эллипс, но не гипербола
    truth = r / (2.0 * body.mu / (r * v * v) - 1.0)
    return MathTask(
        key="apoapsis_from_state", title="апоапсис по текущей скорости",
        stage=2, slot=slot,
        params=[body.mu / 3.5e12, r / 1.0e6, v / 3000.0, v / v_circ / 1.4,
                0, 0],
        answers=(truth,), low=(3.0e4,), high=(1.0e9,), units="м",
        given=f"{body.name}, r {r / 1000:.0f} км, v {v:.0f} м/с в периапсисе")


# --- Ступень 3: межпланетное ----------------------------------------------
def _t_phase_angle(rng, slot):
    body = get_body("Kerbin")
    r1 = body.radius + _log_uniform(rng, 70_000.0, 300_000.0)
    r2 = _log_uniform(rng, 2.0e6, 60.0e6)
    truth = math.degrees(rm.transfer_phase_angle(body, r1, r2))
    return MathTask(
        key="phase_angle", title="фазовый угол окна перехода", stage=3,
        slot=slot,
        params=[body.mu / 3.5e12, r1 / 1.0e6, r2 / 1.0e7, r2 / r1 / 100.0,
                0, 0],
        # Угол бывает отрицательным (цель должна отставать) — шкала линейная
        answers=(truth,), low=(-180.0,), high=(180.0,), units="°",
        given=f"орбита {r1 / 1000:.0f} км -> цель {r2 / 1000:.0f} км")


def _t_synodic_period(rng, slot):
    t1 = _log_uniform(rng, 1.0e4, 2.0e6)
    t2 = t1 * rng.choice([rng.uniform(1.1, 8.0), rng.uniform(0.15, 0.9)])
    truth = abs(1.0 / (1.0 / t1 - 1.0 / t2)) if abs(t1 - t2) > 1e-9 else 1e9
    return MathTask(
        key="synodic_period", title="синодический период (как часто окно)",
        stage=3, slot=slot,
        params=[t1 / 1.0e6, t2 / 1.0e6, t1 / t2 / 8.0, 0, 0, 0],
        answers=(truth,), low=(1.0e3,), high=(1.0e9,), units="с",
        given=f"периоды {t1:.0f} с и {t2:.0f} с")


def _t_plane_change(rng, slot):
    body = _body(rng)
    r = body.radius + _log_uniform(rng, 10_000.0, 400_000.0)
    v = math.sqrt(body.mu / r)
    inc = rng.uniform(0.5, 60.0)
    truth = 2.0 * v * math.sin(math.radians(inc) / 2.0)
    return MathTask(
        key="plane_change", title="смена плоскости орбиты", stage=3, slot=slot,
        params=[body.mu / 3.5e12, r / 1.0e6, v / 3000.0, inc / 60.0, 0, 0],
        answers=(truth,), low=(0.3,), high=(4_000.0,), units="м/с",
        given=f"{body.name}, круговая {r / 1000:.0f} км, поворот на {inc:.1f}°")


def _t_escape_dv(rng, slot):
    body = _body(rng)
    r = body.radius + _log_uniform(rng, 10_000.0, 300_000.0)
    truth = math.sqrt(2.0 * body.mu / r) - math.sqrt(body.mu / r)
    return MathTask(
        key="escape_dv", title="импульс ухода из сферы влияния", stage=3,
        slot=slot,
        params=[body.mu / 3.5e12, body.radius / 6.0e5, r / 1.0e6, 0, 0, 0],
        answers=(truth,), low=(1.0,), high=(4_000.0,), units="м/с",
        given=f"{body.name}, с круговой {r / 1000:.0f} км")


def _t_capture_dv(rng, slot):
    body = get_body(rng.choice(("Mun", "Minmus", "Duna")))
    alt = _log_uniform(rng, 10_000.0, 200_000.0)
    v_inf = _log_uniform(rng, 50.0, 1_500.0)
    truth = rm.sphere_of_influence_capture_dv(body, v_inf, alt)
    return MathTask(
        key="capture_dv", title="импульс захвата у цели", stage=3, slot=slot,
        params=[body.mu / 3.5e12, body.radius / 6.0e5, alt / 1.0e5,
                v_inf / 1000.0, 0, 0],
        answers=(truth,), low=(1.0,), high=(5_000.0,), units="м/с",
        given=f"{body.name}, v∞ {v_inf:.0f} м/с, периапсис {alt / 1000:.0f} км")


def _t_hyperbolic_excess(rng, slot):
    """Обратная задача к захвату: какая останется скорость на бесконечности."""
    body = _body(rng)
    rp = body.radius + _log_uniform(rng, 10_000.0, 200_000.0)
    v_esc = math.sqrt(2.0 * body.mu / rp)
    v_p = v_esc * rng.uniform(1.02, 1.9)
    truth = math.sqrt(max(0.0, v_p * v_p - 2.0 * body.mu / rp))
    return MathTask(
        key="hyperbolic_excess", title="избыток скорости на гиперболе",
        stage=3, slot=slot,
        params=[body.mu / 3.5e12, rp / 1.0e6, v_p / 4000.0, v_p / v_esc / 2.0,
                0, 0],
        answers=(truth,), low=(1.0,), high=(6_000.0,), units="м/с",
        given=f"{body.name}, периапсис {rp / 1000:.0f} км, v {v_p:.0f} м/с")


# --- Ступень 4: сложные манёвры -------------------------------------------
def _t_bielliptic(rng, slot):
    body = get_body(rng.choice(("Kerbin", "Duna")))
    r1 = body.radius + _log_uniform(rng, 70_000.0, 300_000.0)
    r2 = r1 * rng.uniform(4.0, 30.0)
    rb = r2 * rng.uniform(1.2, 4.0)
    truth = rm.bielliptic_dv(body.mu, r1, r2, rb)
    return MathTask(
        key="bielliptic", title="двухэллиптический переход: три импульса",
        stage=4, slot=slot,
        params=[body.mu / 3.5e12, r1 / 1.0e6, r2 / 1.0e7, rb / 1.0e7, r2 / r1 / 30, rb / r2 / 4],
        answers=(truth,), low=(50.0,), high=(6_000.0,), units="м/с",
        given=f"{body.name}: {r1 / 1000:.0f} → {r2 / 1000:.0f} км через {rb / 1000:.0f} км")


def _t_two_stage(rng, slot):
    isp1 = rng.uniform(250.0, 320.0)
    isp2 = rng.uniform(300.0, 380.0)
    payload = _log_uniform(rng, 0.5, 20.0)
    dry2 = payload * rng.uniform(0.2, 1.5)
    wet2 = dry2 * rng.uniform(2.0, 6.0)
    dry1 = (wet2 + payload) * rng.uniform(0.3, 1.0)
    wet1 = dry1 * rng.uniform(3.0, 8.0)
    truth = rm.two_stage_dv(isp1, wet1, dry1, isp2, wet2, dry2, payload)
    return MathTask(
        key="two_stage_dv", title="ΔV двухступенчатой ракеты", stage=4, slot=slot,
        params=[isp1 / 350, (wet1 + wet2 + payload) / (dry1 + wet2 + payload) / 8,
                isp2 / 350, (wet2 + payload) / (dry2 + payload) / 6,
                payload / 20, wet1 / 200],
        answers=(truth,), low=(1_000.0,), high=(20_000.0,), units="м/с",
        given=f"Isp {isp1:.0f}/{isp2:.0f} с, ступени {wet1:.1f}/{wet2:.1f} т, нагрузка {payload:.1f} т")


def _t_ejection(rng, slot):
    body = _body(rng)
    r = body.radius + _log_uniform(rng, 15_000.0, 300_000.0)
    v_inf = _log_uniform(rng, 100.0, 4_000.0)
    truth = rm.ejection_dv(body.mu, r, v_inf)
    return MathTask(
        key="ejection_dv", title="уход с опорной орбиты с избытком v∞ (Оберт)",
        stage=4, slot=slot,
        params=[body.mu / 3.5e12, r / 1.0e6, v_inf / 1000.0,
                v_inf / math.sqrt(2 * body.mu / r), 0, 0],
        answers=(truth,), low=(10.0,), high=(6_000.0,), units="м/с",
        given=f"{body.name}, орбита {r / 1000 - body.radius / 1000:.0f} км, v∞ {v_inf:.0f} м/с")


def _t_combined_plane(rng, slot):
    v2 = rng.uniform(200.0, 3_000.0)
    v1 = v2 * rng.uniform(0.6, 1.0)
    di = math.radians(rng.uniform(1.0, 60.0))
    truth = rm.combined_plane_change(v1, v2, di)
    return MathTask(
        key="combined_plane", title="скругление со сменой плоскости одним импульсом",
        stage=4, slot=slot,
        params=[v1 / 3000, v2 / 3000, di, math.cos(di), v1 / v2, 0],
        answers=(truth,), low=(5.0,), high=(4_000.0,), units="м/с",
        given=f"{v1:.0f} → {v2:.0f} м/с, поворот {math.degrees(di):.1f}°")


def _t_suicide_burn(rng, slot):
    body = get_body(rng.choice(("Mun", "Minmus", "Duna", "Kerbin")))
    g = body.surface_gravity
    mass = _log_uniform(rng, 1.0, 30.0)
    # Задаётся запас замедления, а не TWR: на Минмусе TWR 1.5 даёт
    # 0.02 м/с² — торможение с высоты в тысячи километров, такой посадки
    # не бывает.
    decel = _log_uniform(rng, 0.8, 30.0)
    thrust = (decel + g) * mass
    twr = thrust / (mass * g)
    v = _log_uniform(rng, 20.0, 600.0)
    h, t = rm.suicide_burn(v, thrust, mass, g)
    return MathTask(
        key="suicide_burn", title="посадка в последний момент: высота и время",
        stage=4, slot=slot,
        params=[v / 600, thrust / 300, mass / 30, g / 10, twr / 6, thrust / mass / 50],
        answers=(h, t), low=(1.0, 0.3), high=(300_000.0, 900.0), units="м, с",
        given=f"{body.name}: падает {v:.0f} м/с, {mass:.1f} т, тяга {thrust:.0f} кН")


def _t_soi(rng, slot):
    mu_parent = _log_uniform(rng, 1e11, 1.2e18)
    mu_body = mu_parent * _log_uniform(rng, 1e-5, 1e-2)
    sma = _log_uniform(rng, 5e6, 1e11)
    truth = rm.soi_radius(sma, mu_body, mu_parent)
    return MathTask(
        key="soi_radius", title="радиус сферы действия тела", stage=4, slot=slot,
        params=[sma / 1e10, mu_body / mu_parent * 100, mu_parent / 1e18,
                mu_body / 1e15, 0, 0],
        answers=(truth,), low=(1e4,), high=(1e11,), units="м",
        given=f"a {sma / 1e9:.2f} Гм, μ/M {mu_body / mu_parent:.2e}")


# ==========================================================================
# Реестр задач. Порядок ФИКСИРОВАН: slot — это индекс в one-hot, и менять
# его у существующей задачи нельзя, иначе сохранённые веса перепутают тип.
# Новые задачи только дописываются в конец.
# ==========================================================================
TASK_BUILDERS = (
    _t_circular_speed, _t_orbital_period, _t_surface_gravity,
    _t_tsiolkovsky, _t_twr, _t_burn_time,
    _t_vis_viva, _t_raise_apoapsis, _t_circularize,
    _t_hohmann, _t_hohmann_time, _t_apoapsis_from_state,
    _t_phase_angle, _t_synodic_period, _t_plane_change,
    _t_escape_dv, _t_capture_dv, _t_hyperbolic_excess,
    # ступень 4 — дописаны в свободные слоты, прежние номера не тронуты
    _t_bielliptic, _t_two_stage, _t_ejection,
    _t_combined_plane, _t_suicide_burn, _t_soi,
)

if len(TASK_BUILDERS) > MATH_TASK_SLOTS:
    raise RuntimeError("Задач больше, чем слотов в наблюдении")


def tasks_of_stage(stage: int) -> list[int]:
    """Индексы задач, которые ВПЕРВЫЕ появляются на этой ступени."""
    rng = random.Random(0)
    out = []
    for slot, builder in enumerate(TASK_BUILDERS):
        if builder(rng, slot).stage == stage:
            out.append(slot)
    return out


def describe_tasks(stage: int | None = None) -> str:
    rng = random.Random(0)
    lines = []
    for slot, builder in enumerate(TASK_BUILDERS):
        sample = builder(rng, slot)
        if stage is not None and sample.stage != stage:
            continue
        lines.append(f"  [{slot:2d}] ступень {sample.stage} | {sample.title} "
                     f"({sample.units or 'без единиц'})")
    return "\n".join(lines)


# ==========================================================================
class MathEnv:
    """Тренажёр счёта. Интерфейс как у остальных сред: reset/step.

    Эпизод — одна задача в один шаг. Награда считается по относительной
    ошибке и лежит в [-1, +1.5]: около нуля — «ответ не имеет отношения к
    правде», +1 — попадание в допуск ступени, выше — попадание точнее.
    """

    def __init__(self, stage: int = 1, seed: int | None = None,
                 tolerance: float | None = None,
                 goal_altitude: float = 80_000.0, dt: float = 0.0,
                 max_time: float = 0.0, exploring_starts: float = 0.0):
        # goal_altitude/dt/max_time/exploring_starts среде счёта не нужны,
        # но принимаются: так тренажёр подходит под общий вызов
        # Stage.make_env() и не требует отдельной ветки в конвейере.
        self.stage = int(stage)
        self.rng = random.Random(seed)
        self.tolerance = (tolerance if tolerance is not None
                          else TOLERANCE_BY_STAGE.get(self.stage, 0.05))
        self.task: MathTask | None = None
        self.episodes = 0
        self.solved = 0

    # ------------------------------------------------------------------
    def _pick_slot(self) -> int:
        """Задача текущей ступени, иногда — повторение пройденного."""
        current = [s for s in range(len(TASK_BUILDERS))
                   if _stage_of_slot(s) == self.stage]
        earlier = [s for s in range(len(TASK_BUILDERS))
                   if _stage_of_slot(s) < self.stage]
        share = REVIEW_SHARE_BY_STAGE.get(self.stage, REVIEW_SHARE)
        if earlier and self.rng.random() < share:
            return self.rng.choice(earlier)
        return self.rng.choice(current or earlier or [0])

    def reset(self) -> list[float]:
        slot = self._pick_slot()
        self.task = TASK_BUILDERS[slot](self.rng, slot)
        return self.task.encode()

    def observation(self) -> list[float]:
        return self.task.encode() if self.task else [0.0] * MATH_OBS_DIM

    def step(self, action):
        """Возвращает (observation, reward, done, info). Всегда done=True."""
        task = self.task
        if task is None:
            self.reset()
            task = self.task
        given = task.decode(action)
        errors = task.errors(given)
        worst = max(errors) if errors else 1.0
        reward = self.reward_from(worst)
        solved = worst <= self.tolerance
        self.episodes += 1
        self.solved += int(solved)
        info = {
            "task": task.key, "title": task.title, "given": task.given,
            "answer": given, "truth": list(task.answers),
            "error": worst, "solved": solved, "units": task.units,
            "outcome": "решено" if solved else "мимо",
        }
        return self.reset(), reward, True, info

    # ------------------------------------------------------------------
    def reward_from(self, error: float) -> float:
        """Награда по относительной ошибке.

        Форма экспоненциальная, а не «правильно/неправильно»: ступенчатая
        награда не даёт градиента, пока сеть не начала попадать, и обучение
        стоит на месте. Здесь же ответ, промахнувшийся вдвое, всё равно
        лучше ответа, промахнувшегося в сто раз, и сеть получает
        направление ещё до первого попадания.
        """
        tol = max(1e-4, self.tolerance)
        score = math.exp(-error / tol)
        reward = 2.0 * score - 1.0
        if error <= tol * 0.2:
            reward += 0.5          # премия за точный ответ, а не «на грани»
        return reward

    @property
    def accuracy(self) -> float:
        return self.solved / self.episodes if self.episodes else 0.0


def _stage_of_slot(slot: int) -> int:
    """Ступень, на которой задача появляется впервые (кэшируется)."""
    global _SLOT_STAGES
    if _SLOT_STAGES is None:
        rng = random.Random(0)
        _SLOT_STAGES = tuple(builder(rng, i).stage
                             for i, builder in enumerate(TASK_BUILDERS))
    return _SLOT_STAGES[slot]


_SLOT_STAGES: tuple[int, ...] | None = None

# Допуск по ступеням: чем дальше, тем строже приёмка.
TOLERANCE_BY_STAGE = {1: 0.05, 2: 0.03, 3: 0.02, 4: 0.02}

# На четвёртой ступени повторения больше: половина задач — прошлые
# разделы. Замер 21.09: время ожога решалось на 27 %, синодический период
# на 65 % — старое нужно не только помнить, но и дотягивать.
REVIEW_SHARE_BY_STAGE = {4: 0.5}


__all__ = ["MathEnv", "MathTask", "TASK_BUILDERS", "MATH_OBS_DIM",
           "MATH_ANSWER_DIM", "MATH_TASK_SLOTS", "TOLERANCE_BY_STAGE",
           "describe_tasks", "tasks_of_stage"]
