"""Выбор вычислительного устройства для нейросети KIA.

Здесь нет никакой логики обучения — только честное определение того,
на чём считать градиенты, и подробная диагностика для оператора.

ПОЧЕМУ ПО УМОЛЧАНИЮ ПРОЦЕССОР, А НЕ ВИДЕОКАРТА

Это контринтуитивно, поэтому вот замер (RTX 3070, сеть пилота 78 тыс.
весов, полный вызов `act` с переносом результата обратно в python):

    батч       1      8     16     32    256   решений в секунду
    GPU      442  3 449  7 072 14 006 110 493
    CPU    1 683 12 762 23 143 41 784 241 791

Процессор быстрее на ВСЕХ размерах пакета. Причина не в том, что
видеокарта слабая, а в том, что сеть слишком мала: время уходит не на
умножение матриц, а на запуск ядер и ожидание синхронизации. У GPU этот
расход постоянный — около 2.3 мс на вызов независимо от того, считает
она одно наблюдение или двести пятьдесят шесть.

Симптом, по которому это ловится: видеокарта показывает 100 % загрузки и
при этом почти не греется (35 °C). Она не считает — она ждёт.

Побочная выгода: свободная видеокарта достаётся системе, и всё
остальное на компьютере (видео, браузер) перестаёт подтормаживать.

Переопределить: переменная окружения `KIA_NEURAL_DEVICE=cuda` или `cpu`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..logging_setup import get_logger

log = get_logger("neural.device")

# Сколько потоков разрешено PyTorch. По умолчанию он забирает все ядра и
# компьютер начинает подтормаживать на обычной работе. Два потока дают
# почти всю скорость обучения и оставляют машину живой.
DEFAULT_THREADS = 2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class DeviceInfo:
    available: bool
    name: str
    kind: str                  # "cuda" | "cpu" | "none" — что РЕАЛЬНО работает
    torch_version: str = ""
    cuda_version: str = ""
    total_memory_gb: float = 0.0
    capability: str = ""
    build_has_cuda: bool = False
    gpu_name: str = ""         # видеокарта в системе, даже если не занята

    def describe(self) -> str:
        if not self.available:
            return "PyTorch не установлен — нейросеть работать не будет"
        if self.kind == "cuda":
            return (f"GPU: {self.name} ({self.total_memory_gb:.1f} ГБ, "
                    f"compute {self.capability}), CUDA {self.cuda_version}, "
                    f"torch {self.torch_version}")
        hint = ""
        if not self.build_has_cuda:
            hint = " — установлена CPU-сборка torch, GPU не задействуется"
        elif self.gpu_name:
            hint = (f" | {self.gpu_name} свободна намеренно: на сети такого "
                    f"размера процессор быстрее (замер: kia_neural.py bench)")
        return f"CPU: {self.name}, torch {self.torch_version}{hint}"


def inspect() -> DeviceInfo:
    """Полная диагностика вычислительного устройства."""
    if not TORCH_AVAILABLE:
        return DeviceInfo(available=False, name="—", kind="none")

    build_cuda = bool(getattr(torch.version, "cuda", None))
    has_gpu = torch.cuda.is_available()
    # Показываем то, на чём система РЕАЛЬНО будет считать, а не то, что
    # просто установлено: иначе оператор видит «GPU» в диагностике, а в
    # диспетчере задач — незанятую видеокарту, и не понимает, кто врёт.
    gpu_name = ""
    if has_gpu:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = props.name
        if preferred_kind() == "cuda":
            return DeviceInfo(
                available=True,
                name=props.name,
                kind="cuda",
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda or "?",
                total_memory_gb=props.total_memory / (1024 ** 3),
                capability=f"{props.major}.{props.minor}",
                build_has_cuda=True,
                gpu_name=gpu_name,
            )
    import platform
    return DeviceInfo(
        available=True,
        name=platform.processor() or "CPU",
        kind="cpu",
        torch_version=torch.__version__,
        build_has_cuda=build_cuda,
        gpu_name=gpu_name,
    )


_DEVICE = None
_TUNED = False


def limit_threads(threads: int = DEFAULT_THREADS) -> int:
    """Ограничивает аппетит PyTorch к ядрам процессора.

    Без этого torch забирает все ядра, и компьютер начинает подтормаживать
    на обычной работе — а обучение при этом почти не ускоряется: сеть
    маленькая, выигрыш от лишних потоков съедается их синхронизацией.
    """
    if not TORCH_AVAILABLE:
        return 0
    threads = max(1, int(os.environ.get("KIA_NEURAL_THREADS", threads)))
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (RuntimeError, AttributeError):
        # interop-потоки можно задать только до первой параллельной операции
        pass
    return torch.get_num_threads()


def lower_priority() -> bool:
    """Просит систему считать обучение фоновой задачей.

    Обучение идёт часами, и всё это время человек продолжает пользоваться
    компьютером. Пониженный приоритет означает: как только оператору
    что-то понадобится — видео, браузер, что угодно, — система отдаст
    ядра ему, а обучение подождёт. На скорость это почти не влияет,
    потому что в простое машина всё равно занята только им.
    """
    try:
        if os.name != "nt":
            os.nice(10)                                   # POSIX
            return True
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        return bool(ctypes.windll.kernel32.SetPriorityClass(
            handle, BELOW_NORMAL_PRIORITY_CLASS))
    except Exception as exc:                              # прав может не быть
        log.debug("Приоритет процесса не понижен: %s", exc)
        return False


def tune_runtime() -> None:
    """Однократная настройка среды выполнения под долгое обучение."""
    global _TUNED
    if _TUNED:
        return
    _TUNED = True
    threads = limit_threads()
    polite = lower_priority()
    log.info("Режим долгого счёта: потоков %d, приоритет %s", threads,
             "фоновый" if polite else "обычный")


def preferred_kind() -> str:
    """На чём считать: «cpu» или «cuda». Замеры — в шапке модуля."""
    choice = os.environ.get("KIA_NEURAL_DEVICE", "").strip().lower()
    if choice in ("cpu", "cuda"):
        return choice
    # Сеть пилота — 78 тыс. весов. На такой сети видеокарта проигрывает
    # процессору на любом размере пакета: время уходит на запуск ядер,
    # а не на счёт. Меряется командой `python kia_neural.py bench`.
    return "cpu"


def get_device(prefer_gpu: bool | None = None):
    """torch.device для всех тензоров KIA (кэшируется)."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch не установлен: pip install torch")
    tune_runtime()
    kind = "cuda" if prefer_gpu else preferred_kind()
    if kind == "cuda" and torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        # TF32 ускоряет матричные операции на Ampere (RTX 30xx) без потери
        # качества для наших размеров сети
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        _DEVICE = torch.device("cpu")
    log.info("Устройство обучения: %s (%s)", _DEVICE, inspect().describe())
    if _DEVICE.type == "cpu" and torch.cuda.is_available():
        log.info("Видеокарта свободна намеренно: на сети такого размера она "
                 "медленнее процессора (см. `kia_neural.py bench`). "
                 "Переопределить: KIA_NEURAL_DEVICE=cuda")
    return _DEVICE


def reset_device() -> None:
    """Сбрасывает кэш устройства (нужно после переустановки torch)."""
    global _DEVICE
    _DEVICE = None


def gpu_memory_report() -> str:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return "GPU не используется"
    if _DEVICE is not None and _DEVICE.type != "cuda":
        return "GPU не используется (счёт идёт на процессоре — так быстрее)"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return f"видеопамять: занято {allocated:.0f} МБ, зарезервировано {reserved:.0f} МБ"
