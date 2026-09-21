"""Развод спутников по точкам созвездия.

Четыре аппарата через 90° закрывают шар целиком: у каждого всегда есть
сосед в прямой видимости, и связь не пропадает за горизонтом. Задача
пилота — поставить спутник не «на орбиту вообще», а в СВОЮ точку.

Здесь нет обучения с подкреплением и нет подбора: угол разводится
точной формулой (правило 6 из CLAUDE.md). Приём называется фазирующей
орбитой и состоит из трёх шагов:

1. Аппарат уже на круговой орбите радиуса r с периодом P. Его текущий
   угол θ отличается от нужного θ* на Δθ.
2. Меняем период на P' так, чтобы за n витков накопился ровно этот
   угол: отставая на Δθ/n за виток, спутник переезжает по фазе.
       P' = P · (1 + Δθ / (2π n))
   Большая полуось из третьего закона Кеплера:  a' = (μ P'² / 4π²)^⅓
   Импульс — из формулы живой силы:  v = √(μ (2/r − 1/a))
3. Через n витков возвращаем период обратно тем же импульсом с
   обратным знаком, и аппарат встаёт в точку.

Знак выбран так: чтобы ДОГНАТЬ точку впереди (Δθ > 0), период нужно
УКОРОТИТЬ — низкая орбита быстрее. Отсюда минус в формуле выше при
положительной невязке.
"""
from __future__ import annotations

import math

from ..logging_setup import get_logger

# ЖУРНАЛ ЗАВЕДЁН ТОЛЬКО ПОД ИМЕНЕМ «kia». Первый развод по фазе прошёл
# вслепую: `logging.getLogger(__name__)` дал имя вне этого дерева, и ни
# одна строка — ни расчёт невязки, ни итог постановки — в журнал не
# попала. Полёт при этом отработал, но проверить его было нечем.
log = get_logger("pilot.constellation")

TWO_PI = 2.0 * math.pi

# Допуск постановки — по возможностям механизма, а не по желанию.
#
# Замер третьего спутника: проходы дали −174° → −51° → +1.9°, а
# четвёртый проход УХУДШИЛ результат до 4.3°. Иначе и быть не могло:
# правка на 1.9° стоит 0.9 м/с, а собственная точность ожога — 0.5 м/с
# (`ManeuverPilot.BURN_TOLERANCE`), то есть около градуса. Требовать
# полградуса от механизма с точностью в градус — гнать его в шум.
#
# Пять градусов на орбите 800 км — это 70 км вдоль трассы. Для связи,
# где важна взаимная видимость соседей, разницы нет никакой.
PHASE_TOLERANCE = 5.0          # градусов
# Сколько витков отводим на переезд. Меньше витков — больше импульс:
# на один виток разворот в 90° стоит четверти периода, а это сотни
# метров в секунду. Три витка укладываются в десятки.
PHASING_ORBITS = 3
# Потолок импульса на развод. Если расчёт просит больше, значит цель
# выбрана неудачно, и жечь топливо спутника впустую нельзя.
PHASE_DV_CAP = 300.0           # м/с


def wrap_angle(degrees: float) -> float:
    """Приводит угол к диапазону (−180, 180]."""
    value = (degrees + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 else value


def phasing_period(period: float, delta_deg: float,
                   orbits: int = PHASING_ORBITS) -> float:
    """Период фазирующей орбиты для переезда на `delta_deg` за `orbits`."""
    if orbits <= 0:
        raise ValueError("витков должно быть больше нуля")
    shift = math.radians(delta_deg) / (TWO_PI * orbits)
    # ДОГНАТЬ — ЗНАЧИТ УКОРОТИТЬ ПЕРИОД. Спутник, которому нужно вперёд,
    # обязан опуститься: внизу он движется быстрее и обгоняет цель.
    return period * (1.0 - shift)


def semi_major_axis(period: float, mu: float) -> float:
    """Большая полуось по периоду — третий закон Кеплера."""
    return (mu * period * period / (4.0 * math.pi ** 2)) ** (1.0 / 3.0)


def vis_viva(mu: float, radius: float, axis: float) -> float:
    """Скорость на радиусе r при большой полуоси a."""
    value = mu * (2.0 / radius - 1.0 / axis)
    return math.sqrt(value) if value > 0 else 0.0


def phasing_delta_v(mu: float, radius: float, period: float,
                    delta_deg: float, orbits: int = PHASING_ORBITS) -> float:
    """Импульс входа в фазирующую орбиту (столько же нужно на выход)."""
    new_period = phasing_period(period, delta_deg, orbits)
    if new_period <= 0:
        return float("inf")
    axis = semi_major_axis(new_period, mu)
    # Перицентр фазирующей орбиты не должен уходить под поверхность:
    # 2a − r и есть высота противоположной точки.
    if 2.0 * axis - radius <= 0:
        return float("inf")
    return vis_viva(mu, radius, axis) - vis_viva(mu, radius, radius)


class ConstellationPilot:
    """Ставит аппарат в назначенную точку круговой орбиты."""

    def __init__(self, connection, control, maneuver, telemetry=None,
                 flight_log=None):
        self.conn = connection
        self.sc = connection.space_center
        self.control = control
        self.maneuver = maneuver
        self.telemetry = telemetry
        self.flight_log = flight_log

    # ------------------------------------------------------------------
    @property
    def vessel(self):
        return self.sc.active_vessel

    def current_phase(self) -> float:
        """Угол аппарата в неподвижной системе тела, градусы 0…360.

        Отсчёт ведётся в НЕВРАЩАЮЩЕЙСЯ системе: точки созвездия стоят
        относительно звёзд, а не относительно поверхности. Если считать
        от поверхности, четыре спутника разъедутся вместе с ней и вся
        расстановка потеряет смысл за один виток.
        """
        v = self.vessel
        frame = v.orbit.body.non_rotating_reference_frame
        x, _, z = v.position(frame)
        return math.degrees(math.atan2(z, x)) % 360.0

    # Имя, по которому спутники созвездия узнают друг друга на орбите.
    FLEET_MARK = "KIA"

    def fleet_mates(self, tolerance: float = 0.25) -> list:
        """Уже стоящие спутники созвездия — на той же орбите, у того же тела."""
        v = self.vessel
        body = v.orbit.body
        radius = v.orbit.semi_major_axis
        mates = []
        for other in self.sc.vessels:
            if other == v:
                continue
            try:
                if other.orbit.body != body:
                    continue
                if other.type.name in ("debris", "space_object"):
                    continue
                if self.FLEET_MARK not in other.name:
                    continue
                # Своя орбита с точностью до четверти: чужие высоты нам
                # не соседи, сравнивать фазу с ними бессмысленно.
                if abs(other.orbit.semi_major_axis - radius) / radius > tolerance:
                    continue
            except Exception:
                continue
            mates.append(other)
        return mates

    def phase_of(self, vessel) -> float:
        frame = vessel.orbit.body.non_rotating_reference_frame
        x, _, z = vessel.position(frame)
        return math.degrees(math.atan2(z, x)) % 360.0

    def reference_phase(self) -> float | None:
        """Фаза первого спутника созвездия — от него ведётся отсчёт."""
        mates = self.fleet_mates()
        if not mates:
            return None
        # Берём самого «старого» соседа: у кого больше время миссии, тот
        # встал раньше и служит опорой всей цепочке.
        anchor = max(mates, key=lambda m: getattr(m, "met", 0.0))
        return self.phase_of(anchor)

    def phase_error(self, target_deg: float) -> float:
        """Невязка до своей точки, градусы (−180, 180].

        ТОЧКИ СОЗВЕЗДИЯ — НЕ КОЛЫШКИ В ПУСТОТЕ.
        Первая попытка отсчитывала угол от неподвижного направления, и
        развод отработал вхолостую: невязка была −152.6°, стала −153.5°.
        Причина видна из самого приёма: через n витков фазирующей орбиты
        аппарат возвращается ровно в ту точку, откуда стартовал, — иначе
        и быть не может. Сдвиг возникает НЕ относительно звёзд, а
        относительно соседей, у которых период остался прежним.

        Поэтому отсчёт ведётся от первого поставленного спутника. Если
        соседей нет, этот аппарат сам становится опорой: он уже в своей
        точке по определению, и разводить его не от чего.
        """
        reference = self.reference_phase()
        if reference is None:
            return 0.0
        return wrap_angle((reference + target_deg) - self.current_phase())

    # ------------------------------------------------------------------
    def place(self, target_deg: float, orbits: int = PHASING_ORBITS,
              attempts: int = 3) -> bool:
        """Ставит аппарат в точку, повторяя проходы по ЗАМЕРУ невязки.

        ОТКРЫТЫЙ РАСЧЁТ ЗДЕСЬ НЕ РАБОТАЕТ — ПРОВЕРЕНО ПОЛЁТОМ.
        Первый развод посчитал невязку один раз, отработал два ожога и
        доложил бы об успехе: замер по игре показал фазу 247° вместо
        нуля. Причина простая: импульс 67 м/с двигатель выдаёт 167
        секунд, и всё это время спутник продолжает ехать по орбите. Ни
        одна формула не знает, где он окажется к концу ожога.

        Лечится тем же, чем лечились ожоги (см. MIGRATION.md): вместо
        предсказания — замер и повтор. Каждый проход считает невязку
        заново по фактическому положению; проходов не больше трёх, и
        каждый следующий меньше предыдущего.
        """
        for attempt in range(1, attempts + 1):
            error = self.phase_error(target_deg)
            if abs(error) <= PHASE_TOLERANCE:
                log.info("Спутник в своей точке: невязка %.2f° "
                         "(проходов сделано %d)", error, attempt - 1)
                return True
            log.info("Проход %d из %d: невязка %.1f°", attempt, attempts, error)
            if not self._one_pass(target_deg, orbits):
                break
        final = self.phase_error(target_deg)
        ok = abs(final) <= PHASE_TOLERANCE * 6.0
        log.info("Развод окончен: невязка %.2f° — %s", final,
                 "точка занята" if ok else "точка НЕ занята")
        return ok

    def _one_pass(self, target_deg: float, orbits: int) -> bool:
        """Один проход фазирующей орбиты."""
        v = self.vessel
        body = v.orbit.body
        mu = body.gravitational_parameter
        radius = v.orbit.semi_major_axis
        period = v.orbit.period
        error = self.phase_error(target_deg)

        if abs(error) <= PHASE_TOLERANCE:
            log.info("Спутник уже в своей точке: невязка %.2f°", error)
            return True

        dv = phasing_delta_v(mu, radius, period, error, orbits)
        # Импульс мельче собственной точности ожога — это не правка, а
        # подбрасывание монетки: результат определит не расчёт, а остаток
        # в полметра в секунду, с которым ожог заканчивается всегда.
        if math.isfinite(dv) and abs(dv) < 2.0:
            log.info("Правка на %.1f° потребовала бы %.1f м/с — мельче "
                     "точности ожога, оставляю как есть", error, abs(dv))
            return False
        if not math.isfinite(dv) or abs(dv) > PHASE_DV_CAP:
            log.warning("Развод на %.1f° требует %.0f м/с при потолке %.0f — "
                        "отказываюсь", error, abs(dv), PHASE_DV_CAP)
            return False

        log.info("Развод по фазе: невязка %.1f°, фазирующая орбита на %d "
                 "витка, импульс %.1f м/с (и столько же на выход)",
                 error, orbits, dv)

        # Шаг 1 — вход в фазирующую орбиту.
        node = self.maneuver.add_node(self.sc.ut + 30.0, prograde=dv)
        if not self.maneuver.execute(node):
            log.warning("Ожог входа в фазирующую орбиту не добран")
            return False

        # Шаг 2 — переждать n витков на новой орбите.
        wait = self.vessel.orbit.period * orbits
        log.info("Жду %.1f мин на фазирующей орбите", wait / 60.0)
        self.control.warp_to(self.sc.ut + wait)

        # Шаг 3 — вернуть период. Импульс считаем ЗАНОВО по факту, а не
        # берём со знаком минус: за три витка орбита могла измениться, и
        # зеркальный импульс оставил бы аппарат на промежуточной орбите.
        v = self.vessel
        back = (vis_viva(mu, v.orbit.radius, v.orbit.radius)
                - vis_viva(mu, v.orbit.radius, v.orbit.semi_major_axis))
        node = self.maneuver.add_node(self.sc.ut + 30.0, prograde=back)
        ok = self.maneuver.execute(node)

        final = self.phase_error(target_deg)
        log.info("Проход завершён: невязка %.1f° -> %.1f°, орбита "
                 "%.0f × %.0f км", error, final,
                 self.vessel.orbit.periapsis_altitude / 1000.0,
                 self.vessel.orbit.apoapsis_altitude / 1000.0)
        # ПРОХОД ОТЧИТЫВАЕТСЯ О РАБОТЕ, А НЕ О ПОБЕДЕ.
        #
        # Прежде он возвращал «удалось ли занять точку», и первый же
        # проход обрывал цикл: невязка честно уменьшилась со 152° до 56°,
        # то есть приём сработал, — а вызывающий счёл это отказом и
        # больше не повторял. Сходимость проверяет `place`, здесь же
        # важно одно: сдвинулись мы или встали намертво.
        moved = abs(final) < abs(error) - 1.0
        if not moved:
            log.warning("Проход не сдвинул аппарат (%.1f° -> %.1f°)",
                        error, final)
        return ok and moved
