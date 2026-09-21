"""МОСТ В ИГРУ: как знание математика попадает в живой полёт.

Здесь надо быть точным в формулировках, потому что соблазн соврать
большой. Сеть, обученная в тренажёре математики, НЕ заменяет формулы в
полёте. И не должна: у нас есть точный ответ в `engineer/rocket_math`,
а сеть даёт приближение с ошибкой в проценты. Ставить приближение туда,
где есть точное решение, — значит ухудшать аппарат ради красивой истории
про нейросеть.

Что она делает на самом деле — три вещи, и каждая полезна:

1. НЕЗАВИСИМАЯ ПРОВЕРКА ИСХОДНЫХ ДАННЫХ. Формула считает ровно то, что ей
   дали. Если телеметрия отдала мусор (а она отдаёт: на гиперболической
   траектории KSP присылает апоапсис 1e9 — из-за этого однажды слегло всё
   обучение, см. OBSERVED_APOAPSIS_LIMIT в spaces.py), формула выдаст
   мусор с полной уверенностью. Сеть на тех же входах даст ответ, обученный
   на ФИЗИЧЕСКИ ВОЗМОЖНЫХ задачах, и резкое расхождение — сигнал, что
   входные числа испорчены. Это не второе мнение о математике, это
   детектор непригодных данных.

2. АТТЕСТАЦИЯ. Каждый вызов в живом полёте — экзамен: ответ сети
   сравнивается с точным. Накопленная статистика ложится в
   `kia_math_skills.json` и показывается на пульте. Так видно, какие
   разделы сеть действительно знает, а какие только числится знающей.

3. БЫСТРАЯ ПРИКИДКА ТАМ, ГДЕ ТОЧНОГО РЕШЕНИЯ НЕТ ДЁШЕВО. Поиск окна
   перехода перебирает сотни вариантов, и точный расчёт каждого стоит
   дорого. Отсев заведомо безнадёжных приближением — законная экономия,
   потому что финальный вариант всё равно считается формулой.

Итог: в игре решение принимает формула. Сеть проверяет входные данные и
отвечает за то, чтобы её собственное знание было измеримо.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ROOT
from ..logging_setup import get_logger
from .math_env import TASK_BUILDERS

log = get_logger("neural.math_brain")

SKILLS_FILE = ROOT / "kia_math_skills.json"

# Во сколько раз ответ сети должен разойтись с формулой, чтобы считать
# исходные данные испорченными. Порог намеренно грубый: сеть ошибается на
# проценты, а мусорная телеметрия даёт расхождение в разы и порядки.
GARBAGE_RATIO = 3.0
# Сколько проверок нужно, прежде чем говорить о навыке хоть что-то.
MIN_CHECKS = 20
# Доля попаданий в допуск, начиная с которой раздел считается освоенным.
CONFIRM_RATE = 0.85


@dataclass
class MathSkill:
    """Статистика одного раздела математики по итогам живых проверок."""
    key: str
    checks: int = 0
    within: int = 0
    error_sum: float = 0.0
    worst: float = 0.0
    first_seen: str = ""
    last_seen: str = ""

    @property
    def rate(self) -> float:
        return self.within / self.checks if self.checks else 0.0

    @property
    def mean_error(self) -> float:
        return self.error_sum / self.checks if self.checks else 0.0

    @property
    def confirmed(self) -> bool:
        return self.checks >= MIN_CHECKS and self.rate >= CONFIRM_RATE

    def to_dict(self) -> dict:
        return {"key": self.key, "checks": self.checks, "within": self.within,
                "rate": round(self.rate, 4),
                "mean_error": round(self.mean_error, 5),
                "worst": round(self.worst, 4), "confirmed": self.confirmed,
                "first_seen": self.first_seen, "last_seen": self.last_seen}

    @classmethod
    def from_dict(cls, data: dict) -> "MathSkill":
        skill = cls(key=str(data.get("key", "")))
        skill.checks = int(data.get("checks", 0))
        skill.within = int(data.get("within", 0))
        skill.error_sum = float(data.get("mean_error", 0.0)) * skill.checks
        skill.worst = float(data.get("worst", 0.0))
        skill.first_seen = str(data.get("first_seen", ""))
        skill.last_seen = str(data.get("last_seen", ""))
        return skill


# ==========================================================================
class MathBrain:
    """Обёртка над сетью-математиком для использования в живой игре."""

    def __init__(self, brain=None, tolerance: float = 0.05,
                 path: Path | None = None):
        self.brain = brain
        self.tolerance = float(tolerance)
        self.path = Path(path or SKILLS_FILE)
        self.skills: dict[str, MathSkill] = {}
        self._by_key = {}
        self._agent = None
        self._load()

    # ------------------------------------------------------------------
    def _ensure_agent(self):
        """Сеть подгружается лениво: без неё модуль работает как счётчик."""
        if self._agent is not None:
            return self._agent
        if self.brain is None:
            try:
                from .ppo import build_brain
                self.brain = build_brain()
            except Exception as exc:
                log.warning("Математик недоступен (%s) — проверка отключена", exc)
                return None
        self._agent = getattr(self.brain, "mathematician", None)
        return self._agent

    def _task_index(self):
        """Соответствие «имя задачи -> генератор», строится один раз."""
        if self._by_key:
            return self._by_key
        import random
        rng = random.Random(0)
        for slot, builder in enumerate(TASK_BUILDERS):
            sample = builder(rng, slot)
            self._by_key[sample.key] = (slot, builder)
        return self._by_key

    # ------------------------------------------------------------------
    def predict(self, key: str, **params) -> float | None:
        """Ответ сети на задачу из её программы. None — если не умеет.

        Условие задачи собирается тем же генератором, что и в тренажёре:
        так исключено расхождение в кодировке входа между обучением и
        применением. Генератору подсовывается заранее заданный ответ через
        `params`, а нужен нам только вектор наблюдения.
        """
        agent = self._ensure_agent()
        index = self._task_index()
        if agent is None or key not in index:
            return None
        task = self._build_task(key, params)
        if task is None:
            return None
        try:
            _obs, continuous, _d, _lp, _v = agent.act(
                task.encode(), deterministic=True, update_normalizer=False)
            answer = task.decode([float(v) for v in continuous.squeeze(0).tolist()])
            return float(answer[0])
        except Exception as exc:
            log.debug("Математик не ответил на «%s»: %s", key, exc)
            return None

    def _build_task(self, key: str, params: dict):
        """Собирает объект задачи с готовым условием.

        Генераторы в тренажёре сами разыгрывают условие случайно. Здесь
        условие приходит из полёта, поэтому берётся образец нужного типа и
        у него подменяются числа условия и эталон.
        """
        import copy
        import random

        slot, builder = self._task_index()[key]
        task = builder(random.Random(0), slot)
        supplied = params.get("params")
        if supplied is None:
            return None
        task = copy.replace(task, params=list(supplied),
                            answers=tuple(params.get("answers", task.answers)))
        return task

    # ------------------------------------------------------------------
    def verify(self, key: str, exact: float, params: list[float]) -> dict:
        """Экзамен в полёте: сравнить ответ сети с точным и записать.

        Возвращает словарь с оценкой. РЕШЕНИЕ ПРИНИМАЕТ ВЫЗЫВАЮЩИЙ КОД по
        точному значению — эта функция ничего не подменяет.
        """
        guess = self.predict(key, params=params, answers=(exact,))
        if guess is None:
            return {"key": key, "checked": False, "exact": exact}

        scale = max(abs(exact), 1e-9)
        error = abs(guess - exact) / scale
        ratio = abs(guess / exact) if abs(exact) > 1e-12 else float("inf")
        suspicious = ratio > GARBAGE_RATIO or ratio < 1.0 / GARBAGE_RATIO

        skill = self.skills.get(key) or MathSkill(key=key)
        now = time.strftime("%Y-%m-%d %H:%M")
        skill.checks += 1
        skill.error_sum += error
        skill.worst = max(skill.worst, error)
        skill.within += int(error <= self.tolerance)
        skill.last_seen = now
        if not skill.first_seen:
            skill.first_seen = now
        self.skills[key] = skill

        if suspicious:
            log.warning("Расхождение с математиком по «%s»: формула %.4g, "
                        "сеть %.4g (в %.1f раза). Похоже на испорченные "
                        "исходные данные — проверьте телеметрию.",
                        key, exact, guess, max(ratio, 1.0 / max(ratio, 1e-9)))
        return {"key": key, "checked": True, "exact": exact, "guess": guess,
                "error": error, "within": error <= self.tolerance,
                "suspicious": suspicious, "rate": skill.rate}

    # ------------------------------------------------------------------
    def report(self) -> str:
        if not self.skills:
            return "Математик ещё не проверялся в полёте."
        lines = ["Аттестация математика по живым полётам:", ""]
        for key, skill in sorted(self.skills.items()):
            mark = "ОСВОЕНО" if skill.confirmed else (
                "мало данных" if skill.checks < MIN_CHECKS else "не освоено")
            lines.append(f"  {key:22s} проверок {skill.checks:4d} | "
                         f"в допуске {skill.rate * 100:5.1f} % | "
                         f"средняя ошибка {skill.mean_error * 100:5.2f} % | "
                         f"{mark}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"tolerance": self.tolerance,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "skills": {key: skill.to_dict()
                           for key, skill in sorted(self.skills.items())}}

    def save(self) -> Path:
        self.path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        return self.path

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Не читается %s: %s", self.path.name, exc)
            return
        for key, row in (data.get("skills") or {}).items():
            self.skills[key] = MathSkill.from_dict(row)


__all__ = ["MathBrain", "MathSkill", "SKILLS_FILE"]
