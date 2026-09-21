"""Терминал ASTRAUTOMA: цвета, рамки, клавиши, полноэкранный вывод.

Все цвета — 24-битные ANSI, палитра снята с логотипа: тёмно-синий щит,
серебряная окантовка, оранжевое пламя, зелёная Земля.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time

# --------------------------------------------------------------------------
# Палитра
# --------------------------------------------------------------------------
def rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

SILVER = rgb(206, 212, 222)
STEEL = rgb(128, 140, 160)
NAVY = rgb(58, 92, 150)
SKY = rgb(110, 170, 235)
FLAME = rgb(255, 140, 40)
STAR = rgb(250, 200, 70)
OK = rgb(90, 210, 120)
WARN = rgb(245, 200, 60)
BAD = rgb(240, 80, 70)
MUTED = rgb(100, 108, 124)
WHITE = rgb(240, 244, 250)
SEL_BG = bg(30, 48, 84)

STATUS_COLOR = {"ok": OK, "warn": WARN, "bad": BAD, "info": SKY}
STATUS_MARK = {"ok": "✓", "warn": "▲", "bad": "✗", "info": "•"}

_ANSI = re.compile(r"\033\[[0-9;]*m")


def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - visible_len(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def cut(text: str, width: int) -> str:
    """Обрезает строку по видимой ширине, не ломая ANSI-коды."""
    if visible_len(text) <= width:
        return text
    out, seen, i = [], 0, 0
    while i < len(text) and seen < width - 1:
        m = _ANSI.match(text, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(text[i])
        seen += 1
        i += 1
    return "".join(out) + "…" + RESET


# --------------------------------------------------------------------------
# Консоль
# --------------------------------------------------------------------------
def enable_vt() -> None:
    """Включает ANSI в консоли Windows и UTF-8 на выводе."""
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            handle = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                k32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            os.system("")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def size() -> tuple[int, int]:
    s = shutil.get_terminal_size((120, 40))
    return max(60, s.columns), max(24, s.lines)


def clear() -> None:
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()


def home() -> None:
    sys.stdout.write("\033[H")


def hide_cursor() -> None:
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def alt_screen(on: bool) -> None:
    sys.stdout.write("\033[?1049h" if on else "\033[?1049l")
    sys.stdout.flush()


def draw(lines: list[str]) -> None:
    """Перерисовывает экран без мерцания: курсор домой, строки с очисткой хвоста."""
    w, h = size()
    buf = ["\033[H"]
    for i in range(h - 1):
        line = lines[i] if i < len(lines) else ""
        buf.append(cut(line, w) + RESET + "\033[K\n")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Клавиатура
# --------------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT, ENTER, ESC, BACK, DEL = "up", "down", "left", "right", "enter", "esc", "back", "del"


def read_key(timeout: float | None = None) -> str | None:
    """Одна клавиша. None — если за timeout ничего не нажали."""
    if os.name == "nt":
        import msvcrt
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    code = msvcrt.getwch()
                    return {"H": UP, "P": DOWN, "K": LEFT, "M": RIGHT, "S": DEL}.get(code)
                if ch == "\r":
                    return ENTER
                if ch == "\x1b":
                    return ESC
                if ch == "\x08":
                    return BACK
                if ch == "\x03":
                    raise KeyboardInterrupt
                return ch
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(0.02)
    line = input()
    return ENTER if line == "" else line[0]


def flush_keys() -> None:
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()


# --------------------------------------------------------------------------
# Элементы оформления
# --------------------------------------------------------------------------
def box(title: str, body: list[str], width: int, color: str = STEEL) -> list[str]:
    inner = width - 2
    head = f" {title} " if title else ""
    top = color + "╭─" + RESET + BOLD + SILVER + head + RESET + color + \
        "─" * max(0, inner - 1 - visible_len(head)) + "╮" + RESET
    out = [top]
    for line in body:
        out.append(color + "│" + RESET + " " + pad(cut(line, inner - 2), inner - 2)
                   + " " + color + "│" + RESET)
    out.append(color + "╰" + "─" * inner + "╯" + RESET)
    return out


def bar(fraction: float, width: int, color: str = SKY) -> str:
    fraction = max(0.0, min(1.0, fraction))
    cells = fraction * width
    full = int(cells)
    parts = " ▏▎▍▌▋▊▉"
    rem = parts[int((cells - full) * 8)] if full < width else ""
    empty = width - full - (1 if rem else 0)
    return color + "█" * full + rem + RESET + MUTED + "·" * max(0, empty) + RESET


def status_line(status: str, name: str, text: str, name_w: int = 26) -> str:
    c = STATUS_COLOR.get(status, SILVER)
    return f"{c}{STATUS_MARK.get(status, ' ')} {pad(name, name_w)}{RESET}{c}{text}{RESET}"


def center_block(lines: list[str], width: int) -> list[str]:
    return [pad(l, width, "center") for l in lines]
