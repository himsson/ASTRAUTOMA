"""Two languages: English (default) and Russian.

Every UI string is written as a pair right where it is used:

    L("Mission target", "Цель миссии")

so a translation can never drift away from the text it translates.
"""
from __future__ import annotations

LANGUAGES = {"en": "English", "ru": "Русский"}
LANG = "en"


def set_lang(code: str) -> None:
    global LANG
    LANG = code if code in LANGUAGES else "en"


def L(en: str, ru: str) -> str:
    return ru if LANG == "ru" else en


def ru() -> bool:
    return LANG == "ru"


# Units
def u_ms() -> str:
    return L("m/s", "м/с")


def u_km() -> str:
    return L("km", "км")


def u_t() -> str:
    return L("t", "т")


def u_kn() -> str:
    return L("kN", "кН")


def u_s() -> str:
    return L("s", "с")
