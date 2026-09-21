"""Логотип ASTRAUTOMA из точек.

Картинка assets/logo.jpg (нашивка на чёрном фоне) раскладывается на
шрифт Брайля: каждый символ — это сетка 2×4 точки, поэтому рисунок
получается именно точечным и в 8 раз подробнее обычного ASCII-арта.
Цвет символа — средний цвет его светящихся точек.

Название под нашивкой рисуется так же: текст рендерится широким
шрифтом (как на баннере) и тоже раскладывается на точки.
"""
from __future__ import annotations

from pathlib import Path

from . import term

ROOT = Path(__file__).resolve().parent.parent
LOGO_FILE = ROOT / "assets" / "logo.jpg"
FONTS = ("bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf")

# Биты точек Брайля: (x, y) -> бит
_DOTS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (1, 0): 0x08,
         (1, 1): 0x10, (1, 2): 0x20, (0, 3): 0x40, (1, 3): 0x80}


# Порог для каждой из 8 точек символа (упорядоченный дизеринг): тёмные
# места получают редкие точки, светлые — плотные. Так нашивка выглядит
# именно точечной, а не залитой.
_BAYER = {(0, 0): 0, (1, 2): 1, (1, 0): 2, (0, 2): 3,
          (0, 1): 4, (1, 3): 5, (1, 1): 6, (0, 3): 7}


def _braille(img, threshold: int, tint=None, dither: bool = False) -> list[str]:
    """RGB-картинку с размерами кратными (2, 4) -> цветные строки Брайля."""
    w, h = img.size
    px = img.load()
    lines = []
    for cy in range(0, h, 4):
        row = []
        last = None
        for cx in range(0, w, 2):
            bits, acc, n = 0, [0, 0, 0], 0
            for (dx, dy), bit in _DOTS.items():
                x, y = cx + dx, cy + dy
                if x >= w or y >= h:
                    continue
                r, g, b = px[x, y][:3]
                lum = (r * 299 + g * 587 + b * 114) // 1000
                limit = threshold
                if dither:
                    limit = threshold + _BAYER[(dx, dy)] * 17
                if lum > limit:
                    bits |= bit
                    acc[0] += r
                    acc[1] += g
                    acc[2] += b
                    n += 1
            if not bits:
                row.append(" ")
                continue
            if tint is not None:
                color = tint
            else:
                r, g, b = (min(255, int(c / n * 1.25) + 10) for c in acc)
                color = (r, g, b)
            if color != last:
                row.append(term.rgb(*color))
                last = color
            row.append(chr(0x2800 + bits))
        lines.append("".join(row).rstrip() + term.RESET)
    return lines


def _load_font(px: int):
    from PIL import ImageFont
    for name in FONTS:
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts")):
            path = base / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), px)
                except OSError:
                    continue
    return ImageFont.load_default()


def emblem(cols: int, rows: int) -> list[str]:
    """Нашивка, вписанная в cols×rows символов."""
    try:
        from PIL import Image
    except ImportError:
        return [term.pad(term.SILVER + "[ ASTRAUTOMA ]" + term.RESET, cols, "center")]
    if not LOGO_FILE.exists():
        return []
    img = Image.open(LOGO_FILE).convert("RGB")
    bbox = img.convert("L").point(lambda v: 255 if v > 28 else 0).getbbox()
    if bbox:
        img = img.crop(bbox)
    # У символа высота ≈ 2 ширины, а точек по высоте вдвое больше:
    # в пикселях Брайля соотношение сторон сохраняется как есть.
    max_w, max_h = cols * 2, rows * 4
    scale = min(max_w / img.width, max_h / img.height)
    w = max(2, int(img.width * scale) // 2 * 2)
    h = max(4, int(img.height * scale) // 4 * 4)
    from PIL import ImageEnhance, ImageFilter
    img = img.resize((w, h), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2))
    img = ImageEnhance.Color(img).enhance(1.7)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    return _braille(img, threshold=30, dither=True)


def wordmark(text: str, cols: int, rows: int) -> list[str]:
    """Название точками серебряного цвета."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return [term.pad(term.BOLD + term.SILVER + text + term.RESET, cols, "center")]
    font = _load_font(120)
    probe = Image.new("L", (10, 10))
    d = ImageDraw.Draw(probe)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (r - l + 20, b - t + 20), (0, 0, 0))
    ImageDraw.Draw(img).text((10 - l, 10 - t), text, font=font, fill=(255, 255, 255))
    max_w, max_h = cols * 2, rows * 4
    scale = min(max_w / img.width, max_h / img.height)
    w = max(2, int(img.width * scale) // 2 * 2)
    h = max(4, int(img.height * scale) // 4 * 4)
    img = img.resize((w, h), Image.LANCZOS)
    return _braille(img, threshold=110, tint=(214, 220, 230))


def splash_lines(width: int, height: int) -> list[str]:
    """Полный заставочный экран: нашивка, название, приглашение."""
    word_rows = max(3, min(7, height // 7))
    emblem_rows = max(8, height - word_rows - 6)
    art = emblem(min(width - 4, int(emblem_rows * 2.3)), emblem_rows)
    word = wordmark("ASTRAUTOMA", min(width - 8, word_rows * 12), word_rows)
    block = art + [""] + word
    top = max(0, (height - len(block) - 4) // 2)
    lines = [""] * top + term.center_block(block, width)
    return lines
