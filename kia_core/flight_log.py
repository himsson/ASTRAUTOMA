"""Журнал полёта реального времени: kia_flight_log.md.

Файл пересоздаётся при каждом запуске и обновляется по ходу полёта, чтобы
оператор мог открыть его в любой момент и увидеть:

* ЧТО ИИ ДЕЛАЕТ ПРЯМО СЕЙЧАС — блок «Текущий статус» всегда наверху;
* ЧТО УЖЕ СДЕЛАНО — хронология с отметками реального времени, UT игры и MET.

Файл переписывается целиком на каждом обновлении (он маленький), поэтому
статус наверху всегда актуален, а не теряется в хвосте лога.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .config import ROOT
from .logging_setup import get_logger

log = get_logger("flight_log")

LOG_MD = ROOT / "kia_flight_log.md"
LOG_JSON = ROOT / "kia_flight_log.json"

KIND_ICONS = {
    "status": "▶",
    "fact": "✔",
    "reward": "＋",
    "penalty": "－",
    "warning": "⚠",
    "error": "✖",
    "telemetry": "·",
}


@dataclass
class LogEntry:
    wall_time: str
    ut: float
    met: float
    kind: str
    text: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class FlightLog:
    """Потокобезопасный журнал одного полёта."""

    def __init__(self, md_path: Path | None = None, json_path: Path | None = None):
        self.md_path = Path(md_path or LOG_MD)
        self.json_path = Path(json_path or LOG_JSON)
        self._lock = threading.RLock()
        self.entries: list[LogEntry] = []
        self.header: dict = {}
        self.current_status: str = "ожидание"
        self.status_since: str = ""
        self.finished: bool = False
        self.summary: dict = {}
        self._clock = lambda: (0.0, 0.0)      # источник (UT, MET)

    # ------------------------------------------------------------------
    def retarget(self, md_path: Path, json_path: Path | None = None) -> None:
        """Переводит журнал в папку конкретного запуска (kia/<N>/)."""
        with self._lock:
            self.md_path = Path(md_path)
            self.json_path = Path(json_path or self.md_path.with_suffix(".json"))
            self.md_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def bind_clock(self, fn) -> None:
        """Привязывает источник игрового времени: () -> (ut, met)."""
        self._clock = fn

    def _now(self) -> tuple[str, float, float]:
        try:
            ut, met = self._clock()
        except Exception:
            ut, met = 0.0, 0.0
        return datetime.now().strftime("%H:%M:%S"), float(ut or 0.0), float(met or 0.0)

    # ------------------------------------------------------------------
    def start(self, mission: str, craft: str, design_summary: dict | None = None,
              extra: dict | None = None) -> None:
        """Начинает новый журнал — старый файл затирается."""
        with self._lock:
            self.entries.clear()
            self.finished = False
            self.summary = {}
            self.header = {
                "mission": mission,
                "craft": craft,
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "design": design_summary or {},
            }
            if extra:
                self.header.update(extra)
            self.current_status = "предполётная подготовка"
            self.status_since = datetime.now().strftime("%H:%M:%S")
            self._flush()
            log.info("Журнал полёта создан: %s", self.md_path)

    # ------------------------------------------------------------------
    def status(self, text: str, detail: str = "") -> None:
        """ТЕКУЩИЙ СТАТУС: что ИИ делает прямо сейчас."""
        with self._lock:
            wall, ut, met = self._now()
            self.current_status = text
            self.status_since = wall
            self.entries.append(LogEntry(wall, ut, met, "status", text, detail))
            self._flush()
        log.info("СТАТУС: %s%s", text, f" — {detail}" if detail else "")

    def fact(self, text: str, detail: str = "") -> None:
        """ФАКТ ДЕЙСТВИЯ: что уже произошло."""
        self._add("fact", text, detail)
        log.info("ФАКТ: %s%s", text, f" — {detail}" if detail else "")

    def reward(self, text: str, points: float, detail: str = "") -> None:
        kind = "reward" if points >= 0 else "penalty"
        self._add(kind, f"{text} ({points:+.0f} очков)", detail)

    def warning(self, text: str, detail: str = "") -> None:
        self._add("warning", text, detail)
        log.warning("%s %s", text, detail)

    def error(self, text: str, detail: str = "") -> None:
        self._add("error", text, detail)
        log.error("%s %s", text, detail)

    def telemetry(self, snap) -> None:
        """Разовый срез телеметрии в журнал."""
        detail = (f"h={snap.altitude:.0f} м, Ap={snap.apoapsis:.0f}, "
                  f"Pe={snap.periapsis:.0f}, v={snap.speed:.0f} м/с, "
                  f"m={snap.mass:.2f} т, ступень {snap.stage}")
        self._add("telemetry", f"Телеметрия ({snap.body}, {snap.situation})", detail)

    def _add(self, kind: str, text: str, detail: str = "") -> None:
        with self._lock:
            wall, ut, met = self._now()
            self.entries.append(LogEntry(wall, ut, met, kind, text, detail))
            self._flush()

    # ------------------------------------------------------------------
    def finish(self, summary: dict) -> None:
        with self._lock:
            self.finished = True
            self.summary = summary
            wall, ut, met = self._now()
            self.entries.append(LogEntry(
                wall, ut, met, "fact",
                f"Полёт завершён: {summary.get('total_score', 0):.0f} очков",
                summary.get("termination_reason") or summary.get("final_phase", "")))
            self._flush()
        log.info("Журнал полёта закрыт: %s", self.md_path)

    # ------------------------------------------------------------------
    def tail(self, count: int = 15) -> str:
        with self._lock:
            rows = self.entries[-count:]
            lines = [f"Текущий статус: {self.current_status} (с {self.status_since})"]
            for e in rows:
                icon = KIND_ICONS.get(e.kind, "·")
                detail = f" — {e.detail}" if e.detail else ""
                lines.append(f"  {icon} {e.wall_time} | T+{e.met:6.0f} | {e.text}{detail}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        try:
            self.md_path.write_text(self._render(), encoding="utf-8")
            self.json_path.write_text(json.dumps({
                "header": self.header,
                "current_status": self.current_status,
                "status_since": self.status_since,
                "finished": self.finished,
                "summary": self.summary,
                "entries": [e.to_dict() for e in self.entries],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            log.debug("Журнал не записан: %s", exc)

    def _render(self) -> str:
        h = self.header
        L: list[str] = []
        add = L.append

        add("# Журнал полёта KIA")
        add("")
        add(f"**Задача:** {h.get('mission', '—')}  ")
        add(f"**Аппарат:** {h.get('craft', '—')}  ")
        add(f"**Старт сессии:** {h.get('started_at', '—')}  ")
        design = h.get("design") or {}
        if design:
            add(f"**Расчёт:** масса {design.get('total_mass_t', '—')} т, "
                f"ΔV {design.get('total_delta_v', '—')} м/с, "
                f"TWR {design.get('launch_twr', '—')}, "
                f"деталей {design.get('part_count', '—')}  ")
        add("")

        add("## Текущий статус")
        add("")
        if self.finished:
            add(f"> **ПОЛЁТ ЗАВЕРШЁН** — {self.current_status}")
        else:
            add(f"> **{self.current_status.upper()}** — с {self.status_since}")
        add("")

        if self.summary:
            add(f"**Итог:** {self.summary.get('total_score', 0):.0f} очков | "
                f"достижения: {', '.join(self.summary.get('milestones', [])) or '—'} | "
                f"отказы: {', '.join(self.summary.get('failures', [])) or '—'}")
            add("")

        add("## Хронология")
        add("")
        add("| Время | UT игры | T+ | Событие | Детали |")
        add("|---|---|---|---|---|")
        for e in self.entries:
            icon = KIND_ICONS.get(e.kind, "·")
            text = e.text.replace("|", "/")
            detail = e.detail.replace("|", "/")
            add(f"| {e.wall_time} | {e.ut:.0f} | {e.met:.0f} с | {icon} {text} | {detail} |")
        add("")

        facts = [e for e in self.entries if e.kind in ("fact", "reward", "penalty")]
        if facts:
            add("## Что сделано")
            add("")
            for e in facts:
                detail = f" ({e.detail})" if e.detail else ""
                add(f"- В {e.wall_time} (T+{e.met:.0f} с) {e.text}{detail}")
            add("")
        return "\n".join(L) + "\n"


# Единый журнал текущей сессии
FLIGHT_LOG = FlightLog()
