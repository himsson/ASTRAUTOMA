"""Единая настройка логирования для всех модулей KIA."""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from .config import LOG_DIR

_CONFIGURED = False

# Уровень для сводок дежурного (neural/watchdog.py). Стоит выше INFO,
# чтобы зелёная строка «всё штатно» не терялась в потоке обычных записей,
# но ниже WARNING — это не предупреждение, а отметка «живо и считает».
STATUS = 25
logging.addLevelName(STATUS, "STATUS")


class _ColourFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[36m",
        STATUS: "\033[92m",          # ярко-зелёный: всё идёт как задумано
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        colour = self.COLOURS.get(record.levelno, "")
        return f"{colour}{base}{self.RESET}" if colour else base


def setup_logging(level: int = logging.INFO, session_tag: str | None = None) -> logging.Logger:
    """Инициализирует корневой логгер kia (идемпотентно)."""
    global _CONFIGURED
    logger = logging.getLogger("kia")
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Windows: консоль по умолчанию не UTF-8 — русский текст превращается в мусор
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    tag = session_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"kia_{tag}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(_ColourFormatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                                 datefmt="%H:%M:%S"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.debug("Логирование инициализировано, файл: %s", log_path)
    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"kia.{name}")


__all__ = ["setup_logging", "get_logger", "STATUS"]
