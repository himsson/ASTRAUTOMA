"""Счётчик запросов к игре: что и как часто спрашивает автопилот.

Каждые 5 с пишет в logs/rpc_profile.log: сколько запросов kRPC ушло,
какие самые частые и самые долгие, и темп игрового времени
(1.00 — игра успевает, 0.2 — FPS около 5). Нужен, чтобы искать
тормоза по замеру, а не наугад.
"""
from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "logs" / "rpc_profile.log"
_lock = threading.Lock()
_calls: collections.Counter = collections.Counter()
_time: collections.Counter = collections.Counter()


def install() -> None:
    try:
        from krpc import client
    except ImportError:
        return
    if getattr(client.Client._invoke, "_astra", False):
        return
    original = client.Client._invoke

    def _invoke(self, service, procedure, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, service, procedure, *args, **kwargs)
        finally:
            key = f"{service}.{procedure}"
            with _lock:
                _calls[key] += 1
                _time[key] += time.perf_counter() - t0

    _invoke._astra = True
    client.Client._invoke = _invoke


def start(space_center_getter) -> None:
    """Фоновая запись сводки каждые 5 с."""
    def loop():
        LOG.parent.mkdir(exist_ok=True)
        last_ut, last_t = None, None
        while True:
            time.sleep(5.0)
            try:
                ut = space_center_getter().ut
            except Exception:
                ut = None
            now = time.time()
            pace = ""
            if ut is not None and last_ut is not None:
                pace = f"темп игры {(ut - last_ut) / (now - last_t):.2f}"
            last_ut, last_t = ut, now
            with _lock:
                calls, spent = dict(_calls), dict(_time)
                _calls.clear()
                _time.clear()
            total = sum(calls.values())
            lines = [f"{time.strftime('%H:%M:%S')}  запросов {total / 5:.0f}/с  {pace}"]
            for key, n in sorted(calls.items(), key=lambda kv: -kv[1])[:12]:
                lines.append(f"    {n / 5:6.1f}/с  {spent[key] / n * 1000:6.1f} мс  {key}")
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
    threading.Thread(target=loop, daemon=True).start()
