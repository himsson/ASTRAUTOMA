"""Нейросети ASTRAUTOMA: только загрузка готовых весов (инференс).

Обучение живёт в отдельном приложении (папка Kerbal Intelligence Agency);
сюда веса приходят уже обученными.

    device.py      выбор GPU/CPU
    networks.py    устройство сетей
    ppo.py         загрузка сети из файла весов
    math_brain.py  сеть расчёта манёвров
    solar.py, nav_env.py, nav_plan.py — навигатор межпланетной программы

PyTorch здесь НЕ импортируется при загрузке пакета: модулям планет
(solar, nav_env) он не нужен, а его импорт занимает секунды — меню
подвисало на первом же обращении к списку целей.
"""

__all__ = ["DeviceInfo", "get_device", "inspect", "is_available", "build_brain"]


def is_available() -> bool:
    from .device import TORCH_AVAILABLE
    return TORCH_AVAILABLE


def __getattr__(name):
    if name in ("DeviceInfo", "get_device", "inspect"):
        from . import device
        return getattr(device, name)
    if name == "build_brain":
        from .ppo import build_brain
        return build_brain
    raise AttributeError(name)
