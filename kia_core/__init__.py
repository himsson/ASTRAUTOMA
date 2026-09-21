"""Kerbal Intelligence Agency — автономная агентная ИИ-система для KSP 1 + kRPC.

Модули:
    kia.engineer     — знания о ракетостроении, расчёты, проектирование
    kia.environment  — kRPC: соединение, телеметрия, управление
    kia.pilot        — выведение, манёвры, перелёт к Муне
    kia.learning     — награды, история, эволюционный оптимизатор
    kia.agent        — диспетчер цикла обучения
"""
__version__ = "0.1.0"

from .config import CONFIG
from .logging_setup import get_logger, setup_logging

__all__ = ["CONFIG", "get_logger", "setup_logging", "__version__"]
