"""Замер нагрузки на игру: что именно роняет FPS.

При низком FPS игровое время идёт медленнее реального (KSP ограничивает
шаг физики). Поэтому отношение «игровые секунды / реальные» при 1x —
честный показатель нагрузки: 1.00 — игра успевает, 0.25 — тормозит.

Каждый этап включает ещё одну часть автопилота и меряет 6 секунд.
Ракету не запускает и ничем не управляет, только читает.
Запуск: py -3 tools/fps_probe.py  (KSP в полёте, ракета на столе, kRPC Start Server)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def rate(sc, seconds=6.0) -> float:
    u0, t0 = sc.ut, time.time()
    time.sleep(seconds)
    return (sc.ut - u0) / (time.time() - t0)


def main() -> None:
    import astrautoma
    astrautoma.quiet_logging()
    from kia_core.environment.connection import KRPCConnection
    from kia_core.environment.telemetry import Telemetry, RESOURCE_NAMES

    conn = KRPCConnection().connect()
    sc = conn.space_center
    v = sc.active_vessel
    print(f"Аппарат: {v.name}, деталей {len(v.parts.all)}")
    print(f"1. Только соединение            темп игры {rate(sc):.2f}")

    tel = Telemetry(conn)
    print(f"   потоков открыто: {len(tel._streams)}")
    print(f"2. + потоки телеметрии          темп игры {rate(sc):.2f}")

    # Этап 3: цикл как в выведении, но без команд — только чтение
    import threading
    stop = threading.Event()
    calls = [0]

    def loop():
        while not stop.is_set():
            tel.snapshot()
            calls[0] += 1
            time.sleep(0.05)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    r = rate(sc)
    stop.set()
    th.join()
    print(f"3. + snapshot() 20 раз/с         темп игры {r:.2f}  ({calls[0]/6:.0f} снимков/с)")

    tel.close()
    print(f"4. Потоки закрыты               темп игры {rate(sc):.2f}")

    # Этап 5: только потоки ресурсов — подозреваемый номер один
    streams = []
    for res in RESOURCE_NAMES:
        streams.append(conn.conn.add_stream(v.resources.amount, res))
        streams.append(conn.conn.add_stream(v.resources.max, res))
    print(f"5. Только {len(streams)} потоков ресурсов   темп игры {rate(sc):.2f}")
    for s in streams:
        s.remove()
    conn.close()


if __name__ == "__main__":
    main()
