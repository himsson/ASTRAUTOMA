"""A tilted rocket with a breathing flame for the main menu corner.

The rocket is drawn as a picture (nose cone, body with a window, two
fins, nozzle, flame), rotated, and rendered with braille dots — the same
way as the logo — so the shape reads as a rocket at a glance.
"""
from __future__ import annotations

import math
import random

from . import term as T

W, H = 68, 80          # braille pixels (= 34 × 20 terminal cells)
ANGLE = -40.0          # degrees: nose to the upper right


def _draw(t: float):
    from PIL import Image, ImageDraw
    S = 4                                   # draw big, shrink at the end
    img = Image.new("RGB", (60 * S, 170 * S), (0, 0, 0))
    d = ImageDraw.Draw(img)

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    cx = 30
    silver, dark = (215, 222, 234), (110, 122, 140)
    orange, blue = (255, 125, 35), (120, 185, 250)
    rnd = random.Random(int(t * 10))
    # flame: breathes longer and shorter, the tip flickers
    length = 26 + 14 * (0.5 + 0.5 * math.sin(t * 6.0)) + rnd.uniform(0, 6)
    for layer, (color, half) in enumerate((((240, 70, 30), 9), ((255, 160, 40), 6),
                                           ((255, 240, 150), 3))):
        tip = 118 + length * (1.0 - layer * 0.25)
        d.polygon(P([(cx - half, 116), (cx + half, 116), (cx + rnd.uniform(-2, 2), tip)]), fill=color)
    d.polygon(P([(cx - 6, 108), (cx + 6, 108), (cx + 8, 118), (cx - 8, 118)]), fill=dark)   # nozzle
    d.polygon(P([(cx - 10, 82), (cx - 22, 116), (cx - 10, 108)]), fill=orange)             # fins
    d.polygon(P([(cx + 10, 82), (cx + 22, 116), (cx + 10, 108)]), fill=orange)
    d.rectangle(P([(cx - 10, 42), (cx + 10, 108)]), fill=silver)                           # body
    d.rectangle(P([(cx - 10, 70), (cx + 10, 73)]), fill=dark)                              # stage line
    d.polygon(P([(cx - 10, 42), (cx + 10, 42), (cx, 8)]), fill=orange)                     # nose
    d.ellipse(P([(cx - 6, 48), (cx + 6, 60)]), fill=blue, outline=dark, width=2 * S)       # window
    img = img.rotate(ANGLE, resample=Image.BICUBIC, expand=True)
    img = img.crop(img.getbbox() or (0, 0, img.width, img.height))
    scale = min(W / img.width, H / img.height)
    size = (max(2, int(img.width * scale) // 2 * 2), max(4, int(img.height * scale) // 4 * 4))
    return img.resize(size, Image.LANCZOS)


def frame(t: float) -> list[str]:
    try:
        from .logo import _braille
        img = _draw(t)
    except Exception:
        return []
    return _braille(img, threshold=40)


def overlay(lines: list[str], art: list[str], column: int, top: int = 0) -> list[str]:
    """Places art to the right of existing lines starting at `column`."""
    out = list(lines)
    while len(out) < top + len(art):
        out.append("")
    for i, piece in enumerate(art):
        row = top + i
        if T.visible_len(out[row]) < column:
            out[row] = T.pad(out[row], column) + piece
    return out
