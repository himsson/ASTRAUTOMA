"""Точка входа ASTRAUTOMA. Запуск: ASTRAUTOMA.bat или `python astrautoma.py`.

    --demo   прогон интерфейса без игры (таблица автопилота на искусственных данных)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def quiet_logging() -> None:
    """Журнал пилотов — только в файл logs/: консоль занята интерфейсом."""
    (ROOT / "logs").mkdir(exist_ok=True)
    from kia_core.logging_setup import setup_logging
    logger = setup_logging()
    for handler in list(logger.handlers):
        if type(handler) is logging.StreamHandler:
            logger.removeHandler(handler)


def main() -> int:
    quiet_logging()
    from astra import rpcprofile
    rpcprofile.install()
    from astra.app import App
    return App(demo="--demo" in sys.argv).run()


if __name__ == "__main__":
    sys.exit(main())
