"""Совместная работа процессора и видеокарты на выборе действий.

ЗАЧЕМ. После векторизации расклад времени такой: сеть 21-27 %, физика
73-79 %. Физика — чистый python и живёт на одном ядре, ускорить её
переносом на видеокарту нельзя. А вот сеть можно поделить: часть пачки
считает процессор, часть — видеокарта, одновременно.

КАК ДЕЛИТСЯ. Процессору достаётся БОЛЬШАЯ доля: на сети такого размера
он быстрее (замер в `kia_neural.py bench`). Видеокарта берёт меньшую
часть и считает её параллельно, в отдельном потоке — операции CUDA
отпускают GIL, поэтому это настоящая одновременность, а не чередование.

ЧЕСТНО О ЦЕНЕ — ЗАМЕР, А НЕ ОЦЕНКА.

    доля GPU 0.00: 11 623 шаг/с   только процессор
    доля GPU 0.35:  6 161 шаг/с   62 % процессор / 38 % видеокарта
    доля GPU 0.50:  6 259 шаг/с   поровну

Совместный режим стоит примерно 45 % скорости, и это НЕ ЛЕЧИТСЯ. Была
попытка спрятать работу видеокарты за расчётом физики (см. `while_waiting`
ниже): пока GPU считает свою долю, процессор гоняет физику своей. Не
помогло, и вот почему: 2.3 мс видеокарты — это не счёт на устройстве, а
python-обвязка запуска двух десятков мелких ядер, и она держит GIL.
Настоящей одновременности не получается, потоки выстраиваются в очередь.

Поэтому по умолчанию режим ВЫКЛЮЧЕН. Включать имеет смысл, если хочется
видеть загрузку видеокарты и не жалко почти половины скорости обучения:

    KIA_NEURAL_GPU_SHARE=0.0    только процессор (по умолчанию, быстрее)
    KIA_NEURAL_GPU_SHARE=0.35   65 % процессор, 35 % видеокарта
    KIA_NEURAL_GPU_SHARE=0.5    поровну

В пульте это пункт «Совместный счёт CPU+GPU» подменю нейросети.

ЗЕРКАЛО ВЕСОВ. Обучается ОДНА сеть — та, что на процессоре. Копия на
видеокарте только считает и синхронизируется после каждого обновления
весов. Так исключён самый неприятный класс ошибок: две расходящиеся
копии, обучающиеся каждая по-своему.
"""
from __future__ import annotations

import copy
import os
import threading

from ..logging_setup import get_logger

log = get_logger("neural.hybrid")

# Ноль: по умолчанию считает только процессор. Замер в шапке модуля
# показывает, что любое участие видеокарты стоит около 45 % скорости.
DEFAULT_GPU_SHARE = 0.0
SETTING_FILE = "gpu_share"          # data/gpu_share — выбор оператора
# Меньше этого числа наблюдений отдавать видеокарте бессмысленно: её
# постоянный расход на вызов съест весь смысл
MIN_GPU_BATCH = 4


def _share_file():
    from ..config import DATA_DIR
    return DATA_DIR / SETTING_FILE


def configured_share() -> float:
    """Доля видеокарты: переменная окружения, затем выбор оператора."""
    raw = os.environ.get("KIA_NEURAL_GPU_SHARE")
    if raw is None:
        try:
            raw = _share_file().read_text(encoding="utf-8").strip()
        except OSError:
            return DEFAULT_GPU_SHARE
    try:
        return max(0.0, min(0.9, float(raw)))
    except (ValueError, TypeError):
        return DEFAULT_GPU_SHARE


def save_share(share: float) -> float:
    """Запоминает выбор оператора между запусками пульта."""
    share = max(0.0, min(0.9, float(share)))
    path = _share_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{share:.2f}", encoding="utf-8")
    return share


class HybridActor:
    """Выбор действий силами процессора и видеокарты одновременно."""

    def __init__(self, master, share: float | None = None):
        import torch

        self.torch = torch
        self.master = master                 # обучаемая сеть (процессор)
        self.share = configured_share() if share is None else share
        self.mirror = None
        self.device = None
        self.calls = 0
        self.gpu_decisions = 0
        self.cpu_decisions = 0

        if self.share <= 0.0 or not torch.cuda.is_available():
            return
        if next(master.parameters()).device.type != "cpu":
            # мастер уже на видеокарте — делить нечего
            return
        try:
            self.device = torch.device("cuda")
            self.mirror = copy.deepcopy(master).to(self.device).eval()
            for parameter in self.mirror.parameters():
                parameter.requires_grad_(False)
            log.info("Гибридный режим: %.0f %% пачки считает видеокарта, "
                     "%.0f %% процессор", self.share * 100,
                     (1.0 - self.share) * 100)
        except Exception as exc:
            log.warning("Гибридный режим не включился: %s", exc)
            self.mirror = None
            self.device = None

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mirror is not None

    def sync(self) -> None:
        """Обновляет зеркало после шага обучения."""
        if self.mirror is None:
            return
        self.mirror.load_state_dict(self.master.state_dict())

    def split(self, total: int) -> int:
        """Сколько наблюдений отдать видеокарте."""
        if not self.enabled or total < MIN_GPU_BATCH * 2:
            return 0
        count = int(round(total * self.share))
        if count < MIN_GPU_BATCH:
            return 0
        return min(count, total - MIN_GPU_BATCH)   # процессору тоже оставить

    # ------------------------------------------------------------------
    def act(self, obs, mask=None, while_waiting=None):
        """Действия для всей пачки. obs и mask — тензоры на процессоре.

        while_waiting — работа, которую процессор успевает сделать, ПОКА
        видеокарта считает свою долю. Через неё прячется главная беда
        совместного режима: видеокарта тратит на вызов около 2.3 мс
        независимо от размера пачки, и без такого перекрытия шаг просто
        ждал бы её. Тренер передаёт сюда расчёт физики для своей половины
        аппаратов — он всё равно занимает три с лишним миллисекунды.

        Возвращает (действия, результат while_waiting).
        """
        torch = self.torch
        total = int(obs.shape[0])
        gpu_count = self.split(total)
        if gpu_count <= 0:
            actions = self.master.act(obs, False, mask)
            return actions, (while_waiting(actions) if while_waiting else None)

        cpu_count = total - gpu_count
        self.calls += 1
        self.gpu_decisions += gpu_count
        self.cpu_decisions += cpu_count

        # Зеркало должно нормировать вход теми же числами, что и мастер:
        # статистика нормализатора живёт в буферах и меняется каждый шаг
        normalizer = self.mirror.normalizer
        source = self.master.normalizer
        normalizer.mean.copy_(source.mean.to(self.device, non_blocking=True))
        normalizer.var.copy_(source.var.to(self.device, non_blocking=True))
        normalizer.count.copy_(source.count.to(self.device, non_blocking=True))

        gpu_obs = obs[cpu_count:].to(self.device, non_blocking=True)
        gpu_mask = (mask[cpu_count:].to(self.device, non_blocking=True)
                    if mask is not None else None)
        result: dict = {}

        def on_gpu() -> None:
            try:
                with torch.no_grad():
                    result["value"] = self.mirror.act(gpu_obs, False, gpu_mask)
            except Exception as exc:                  # не ронять обучение
                result["error"] = exc

        worker = threading.Thread(target=on_gpu, name="kia-gpu-actor",
                                  daemon=True)
        worker.start()
        cpu_part = self.master.act(obs[:cpu_count], False,
                                   None if mask is None else mask[:cpu_count])
        # Пока видеокарта считает — процессор занят полезным делом, а не
        # ожиданием. Именно здесь совместный режим перестаёт быть дороже
        # одиночного.
        waited = while_waiting(cpu_part) if while_waiting else None
        worker.join()

        if "error" in result:
            log.warning("Видеокарта отвалилась (%s) — дальше только процессор",
                        result["error"])
            self.mirror = None
            gpu_part = self.master.act(obs[cpu_count:], False,
                                       None if mask is None else mask[cpu_count:])
        else:
            gpu_part = tuple(item.to("cpu") for item in result["value"])

        merged = tuple(torch.cat([left, right], dim=0)
                       for left, right in zip(cpu_part, gpu_part))
        return merged, waited

    def cpu_count(self, total: int) -> int:
        """Сколько наблюдений останется процессору (для перекрытия работы)."""
        return total - self.split(total)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        if not self.enabled:
            return "гибридный режим выключен: считает только процессор"
        total = self.cpu_decisions + self.gpu_decisions
        if total <= 0:
            return (f"гибридный режим: {(1 - self.share) * 100:.0f} % "
                    f"процессор / {self.share * 100:.0f} % видеокарта")
        return (f"решений посчитано: процессор {self.cpu_decisions} "
                f"({self.cpu_decisions / total:.0%}), видеокарта "
                f"{self.gpu_decisions} ({self.gpu_decisions / total:.0%})")


__all__ = ["HybridActor", "configured_share", "DEFAULT_GPU_SHARE"]
