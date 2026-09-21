"""Подключение к серверу kRPC внутри KSP."""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("env.connection")

try:
    import krpc
    from krpc.error import RPCError, ConnectionError as KrpcConnectionError
    KRPC_AVAILABLE = True
except ImportError:  # pragma: no cover
    krpc = None
    RPCError = Exception
    KrpcConnectionError = Exception
    KRPC_AVAILABLE = False


class ConnectionFailed(RuntimeError):
    """Не удалось подключиться к kRPC."""


@dataclass
class ServerInfo:
    version: str
    game_scene: str
    vessel_name: str | None
    ut: float

    def describe(self) -> str:
        return (f"kRPC v{self.version} | сцена: {self.game_scene} | "
                f"судно: {self.vessel_name or '—'} | UT={self.ut:.0f}")


class KRPCConnection:
    """Обёртка над krpc.connect с ретраями и удобным доступом к сервисам."""

    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG.connection
        self.conn = None

    # ------------------------------------------------------------------
    def connect(self) -> "KRPCConnection":
        if not KRPC_AVAILABLE:
            raise ConnectionFailed("Пакет krpc не установлен: pip install krpc")
        last_error: Exception | None = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                log.info("Подключение к kRPC %s:%s (попытка %d/%d)...",
                         self.cfg.address, self.cfg.rpc_port, attempt,
                         self.cfg.retry_attempts)
                self.conn = krpc.connect(
                    name=self.cfg.name,
                    address=self.cfg.address,
                    rpc_port=self.cfg.rpc_port,
                    stream_port=self.cfg.stream_port,
                )
                info = self.info()
                log.info("Соединение установлено. %s", info.describe())
                return self
            except Exception as exc:  # сеть/таймаут/отказ сервера
                last_error = exc
                log.warning("Не удалось подключиться: %s", exc)
                if attempt < self.cfg.retry_attempts:
                    time.sleep(self.cfg.retry_delay)
        raise ConnectionFailed(
            f"kRPC недоступен на {self.cfg.address}:{self.cfg.rpc_port}. "
            f"Запустите KSP, откройте окно kRPC и нажмите Start Server. "
            f"Последняя ошибка: {last_error}")

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
                log.info("Соединение с kRPC закрыто")
            except Exception as exc:
                log.debug("Ошибка при закрытии соединения: %s", exc)
            finally:
                self.conn = None

    # ------------------------------------------------------------------
    @property
    def space_center(self):
        self._require()
        return self.conn.space_center

    @property
    def active_vessel(self):
        return self.space_center.active_vessel

    def info(self) -> ServerInfo:
        self._require()
        status = self.conn.krpc.get_status()
        scene = str(self.conn.krpc.current_game_scene)
        vessel_name = None
        ut = 0.0
        try:
            sc = self.conn.space_center
            ut = sc.ut
            vessel_name = sc.active_vessel.name
        except Exception:
            pass
        return ServerInfo(version=status.version, game_scene=scene,
                          vessel_name=vessel_name, ut=ut)

    def is_alive(self) -> bool:
        """Жива ли связь. Не чаще раза в 2 с и дешёвым запросом.

        Раньше здесь был KRPC.GetStatus — 7.5 мс на стороне игры, и цикл
        выведения звал проверку 16 раз в секунду (замер rpc_profile.log).
        Чтение игрового времени стоит 0.1 мс и так же падает без связи.
        """
        if self.conn is None:
            return False
        now = time.monotonic()
        if now - getattr(self, "_alive_checked", 0.0) < 2.0:
            return getattr(self, "_alive", True)
        self._alive_checked = now
        try:
            self.conn.space_center.ut
            self._alive = True
        except Exception:
            self._alive = False
        return self._alive

    def in_flight(self) -> bool:
        """Полётная сцена? Сравнение без учёта регистра: разные сборки kRPC
        возвращают то `GameScene.Flight`, то `GameScene.flight`."""
        try:
            return "flight" in str(self.conn.krpc.current_game_scene).lower()
        except Exception:
            return False

    def game_scene(self) -> str:
        try:
            return str(self.conn.krpc.current_game_scene).split(".")[-1]
        except Exception:
            return "unknown"

    def _require(self) -> None:
        if self.conn is None:
            raise ConnectionFailed("Соединение не установлено — вызовите connect()")

    # ------------------------------------------------------------------
    def __enter__(self) -> "KRPCConnection":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def quick_check() -> ServerInfo:
    """Разовая проверка доступности сервера (используется init-скриптом)."""
    conn = KRPCConnection()
    try:
        conn.connect()
        return conn.info()
    finally:
        conn.close()
