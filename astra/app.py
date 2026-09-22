"""ASTRAUTOMA — main application: language, splash, menu, screens."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from . import analysis, autopilot, craft, designer
from . import i18n
from . import knowledge as K
from . import logo as LG
from . import mission as ms
from . import relays as RL
from . import rocket as RK
from . import term as T
from . import world as W
from .i18n import L

ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.1.0"


def fmt(v: float) -> str:
    return analysis.fmt(v)


# ==========================================================================
class App:
    def __init__(self, demo: bool = False):
        self.settings = W.load_settings()
        i18n.set_lang(self.settings.get("language", "en"))
        self.world: W.World | None = None
        self.vessel: craft.Vessel | None = None
        self.target_code = self.settings.get("target", "LND")
        self.demo = demo
        self.loading = ""
        self.design_cfg = designer.DesignConfig(payload=float(self.settings.get("payload_t", 1.0)))

    # --- state -------------------------------------------------------------
    def detect(self) -> None:
        self.loading = L("Looking for the save and connecting to kRPC…",
                         "Поиск сохранения и подключение к kRPC…")
        self.world = W.detect(self.settings, connect=not self.demo)
        self.loading = L("Reading the loaded blueprint…", "Чтение загруженного чертежа…")
        self.vessel = craft.loaded(self.world)
        self.loading = L("Loading mission targets…", "Загрузка целей полёта…")
        self.available_targets()               # planets and weights — read once, here
        self.loading = ""
        self._budgets = {}
        threading.Thread(target=self._prefill_budgets, daemon=True).start()

    def _prefill_budgets(self) -> None:
        """Δv budgets of every unlocked target, computed in the background."""
        for t in self.available_targets():
            key = (t.code, i18n.LANG)
            if key not in self._budgets:
                try:
                    self._budgets[key] = ms.budget(self.world, t)
                except Exception:
                    pass

    def budget_of(self, t: ms.Target) -> ms.Budget:
        cache = getattr(self, "_budgets", {})
        key = (t.code, i18n.LANG)
        if key not in cache:
            cache[key] = ms.budget(self.world, t)
        return cache[key]

    def save_settings(self) -> None:
        try:
            W.SETTINGS_FILE.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        except OSError:
            pass

    def available_targets(self) -> list[ms.Target]:
        unlocked = K.unlocked_targets()
        return [t for t in ms.targets(self.world) if t.code in unlocked]

    @property
    def target(self) -> ms.Target | None:
        items = self.available_targets()
        return next((t for t in items if t.code == self.target_code), items[0] if items else None)

    def save_target(self, code: str) -> None:
        self.target_code = code
        self.settings["target"] = code
        self.save_settings()

    # --- layout ------------------------------------------------------------
    def header(self, section: str) -> list[str]:
        w, _ = T.size()
        wd = self.world
        left = f" {T.BOLD}{T.SILVER}✦ ASTRAUTOMA{T.RESET}{T.STEEL}  ·  {section}{T.RESET}"
        right = f"{T.MUTED}{datetime.now().strftime('%H:%M:%S')}{T.RESET} "
        link = f"{T.OK}● kRPC{T.RESET}" if wd and wd.live else f"{T.BAD}○ kRPC{T.RESET}"
        top = T.pad(left, w - T.visible_len(right) - 10) + link + "   " + right
        save = wd.save_name if wd and wd.save_name else "—"
        ship = self.vessel.name if self.vessel else "—"
        tgt = self.target
        target_txt = (f"{T.SKY}{tgt.code}{T.RESET} {T.WHITE}{tgt.title}{T.RESET}" if tgt
                      else f"{T.BAD}{L('no flight weights', 'нет весов полёта')}{T.RESET}")
        line2 = (f" {T.MUTED}{L('save', 'сохранение')}{T.RESET} {T.WHITE}{save}{T.RESET}"
                 f"   {T.MUTED}{L('blueprint', 'чертёж')}{T.RESET} {T.WHITE}{ship}{T.RESET}"
                 f"   {T.MUTED}{L('target', 'цель')}{T.RESET} {target_txt}")
        return [top, line2, " " + T.STEEL + "─" * (w - 2) + T.RESET]

    def screen(self, section: str, body: list[str], footer: str, offset: int = 0) -> int:
        """Draws a screen; long bodies scroll. Returns how many lines fit."""
        _, h = T.size()
        room = max(5, h - 6)
        visible = body[offset:offset + room]
        lines = self.header(section) + [""] + visible
        lines += [""] * max(0, h - 2 - len(lines))
        more = ""
        if len(body) > room:
            more = L(f"   ↑↓ scroll {offset + 1}–{min(len(body), offset + room)} of {len(body)}",
                     f"   ↑↓ прокрутка {offset + 1}–{min(len(body), offset + room)} из {len(body)}")
        lines.append(" " + T.MUTED + footer + more + T.RESET)
        T.draw(lines)
        return room

    def wait_key(self, section: str, body: list[str], footer: str | None = None) -> str:
        footer = footer or L("Enter / Esc — back", "Enter / Esc — назад")
        offset = 0
        while True:
            room = self.screen(section, body, footer, offset)
            key = T.read_key(1.0)
            if key == T.DOWN and offset + room < len(body):
                offset += 1
            elif key == T.UP and offset > 0:
                offset -= 1
            elif key is not None and key not in (T.UP, T.DOWN):
                return key

    def no_target(self, section: str) -> bool:
        """True (and a message) when no flight weights unlock any target."""
        if self.target is not None:
            return False
        self.wait_key(section, [
            f"  {T.BAD}✗ {L('No mission targets available.', 'Нет доступных целей полёта.')}{T.RESET}", "",
            "  " + L("Targets are unlocked by installed weights (the FlightProfile module).",
                     "Цели открывают установленные веса (модуль FlightProfile)."),
            "  " + L("Open “Weights / Knowledge” and install a flight profile.",
                     "Откройте «Веса / Знания» и установите профиль полёта.")])
        return True

    # ======================================================================
    # Background work with a visible step list
    # ======================================================================
    def work(self, section: str, steps: list) -> list | None:
        state = {"i": 0, "frac": 0.0, "results": [], "error": None, "done": False}

        def progress(f):
            state["frac"] = max(0.0, min(1.0, float(f)))

        def runner():
            try:
                for i, (_, fn) in enumerate(steps):
                    state["i"], state["frac"] = i, 0.0
                    state["results"].append(fn(progress))
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
            state["done"] = True

        threading.Thread(target=runner, daemon=True).start()
        t0 = time.time()
        while True:
            spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int((time.time() - t0) * 10) % 10]
            body = []
            for i, (label, _) in enumerate(steps):
                if i < state["i"] or (state["done"] and not state["error"]):
                    body.append(f"   {T.OK}✓{T.RESET} {T.SILVER}{label}{T.RESET}")
                elif i == state["i"]:
                    if state["error"]:
                        body.append(f"   {T.BAD}✗ {label}{T.RESET}")
                    else:
                        pct = (f"  {T.bar(state['frac'], 20)} {int(state['frac'] * 100)}%"
                               if state["frac"] > 0 else "")
                        body.append(f"   {T.FLAME}{spin}{T.RESET} {T.WHITE}{T.BOLD}{label}…{T.RESET}{pct}")
                else:
                    body.append(f"   {T.MUTED}○ {label}{T.RESET}")
            if state["error"]:
                body += ["", f"   {T.BAD}{state['error']}{T.RESET}"]
                self.wait_key(section, body)
                return None
            if state["done"]:
                return state["results"]
            self.screen(section, body, L(f"working… {time.time() - t0:4.1f} s",
                                         f"работаю… {time.time() - t0:4.1f} с"))
            time.sleep(0.08)

    # ======================================================================
    # Language (very first start) and splash
    # ======================================================================
    def choose_language(self, first: bool = False) -> None:
        codes = list(i18n.LANGUAGES)
        sel = codes.index(self.settings.get("language", "en")) if not first else 0
        while True:
            w, h = T.size()
            lines = [""] * max(0, h // 2 - 5)
            lines.append(T.pad(f"{T.BOLD}{T.SILVER}ASTRAUTOMA{T.RESET}", w, "center"))
            lines.append("")
            lines.append(T.pad(f"{T.STEEL}Choose your language · Выберите язык{T.RESET}", w, "center"))
            lines.append("")
            for i, code in enumerate(codes):
                name = i18n.LANGUAGES[code]
                if i == sel:
                    item = f"{T.SEL_BG}{T.FLAME} ▸ {T.WHITE}{T.BOLD}{T.pad(name, 14)}{T.RESET}"
                else:
                    item = f"   {T.SILVER}{T.pad(name, 14)}{T.RESET}"
                lines.append(T.pad(item, w, "center"))
            lines += ["", T.pad(f"{T.MUTED}↑↓ · Enter{T.RESET}", w, "center")]
            T.draw(lines)
            key = T.read_key(1.0)
            if key == T.UP:
                sel = (sel - 1) % len(codes)
            elif key == T.DOWN:
                sel = (sel + 1) % len(codes)
            elif key == T.ENTER:
                self.settings["language"] = codes[sel]
                i18n.set_lang(codes[sel])
                self.save_settings()
                return
            elif key == T.ESC and not first:
                return

    def splash(self) -> None:
        """Logo while the save, kRPC and the blueprint load. Enter before the
        loading is done does not freeze the screen: a spinner runs until
        everything is ready, then the menu opens by itself."""
        def load():
            self.detect()
            self._corner_logo()                 # the menu's logo, drawn ahead of time
        worker = threading.Thread(target=load, daemon=True)
        worker.start()
        cache = {}
        blink = 0
        entering = False
        t0 = time.time()
        while True:
            w, h = T.size()
            if cache.get("size") != (w, h):
                cache["size"] = (w, h)
                cache["art"] = LG.splash_lines(w, h - 3)
            lines = list(cache["art"])
            lines += [""] * max(0, h - 4 - len(lines))
            busy = worker.is_alive()
            spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int((time.time() - t0) * 10) % 10]
            if busy:
                msg = f"{T.FLAME}{spin}{T.RESET} {T.MUTED}{self.loading or L('Loading…', 'Загрузка…')}{T.RESET}"
                prompt = (f"{T.STEEL}{L('Loading, the menu opens by itself…', 'Загрузка, меню откроется само…')}{T.RESET}"
                          if entering else f"{T.MUTED}{L('Press Enter', 'Нажмите Enter')}{T.RESET}")
            else:
                if entering:
                    return
                wd = self.world
                bits = [L(f"save “{wd.save_name}”", f"сохранение «{wd.save_name}»") if wd.save_name
                        else L("no save found", "сохранение не найдено"),
                        L("kRPC connected", "kRPC подключён") if wd.live
                        else L("kRPC not connected", "kRPC не подключён")]
                msg = f"{T.STEEL}{'  ·  '.join(bits)}{T.RESET}"
                prompt = (f"{T.WHITE}{T.BOLD}{L('Press Enter', 'Нажмите Enter')}{T.RESET}" if blink % 5 < 3
                          else f"{T.STEEL}{L('Press Enter', 'Нажмите Enter')}{T.RESET}")
            lines = lines[:h - 3] + [T.pad(msg, w, "center"), "", T.pad(prompt, w, "center")]
            T.draw(lines)
            key = T.read_key(0.1)
            blink += 1
            if key == T.ENTER:
                entering = True
                if not worker.is_alive():
                    return
            if key == T.ESC:
                raise SystemExit(0)

    # ======================================================================
    # Menu
    # ======================================================================
    def menu_items(self) -> list[tuple[str, str, str]]:
        return [
            ("target", L("Mission target", "Цель миссии"),
             L("pick an orbit or a landing", "выбор орбиты или посадки")),
            ("analysis", L("Pre-flight analysis", "Предполётный анализ"),
             L("is the loaded blueprint enough for the target", "хватит ли загруженного чертежа на цель")),
            ("autopilot", L("Automatic flight", "Автоматический полёт"),
             L("build the plan and fly it", "расчёт плана и полёт по нему")),
            ("design", L("Rocket design", "Проектирование ракеты"),
             L("a rocket for the target, with real parts", "ракета под цель, из настоящих деталей")),
            ("weights", L("Weights / Knowledge", "Веса / Знания"),
             L("what the AI knows: install and remove", "что знает ИИ: установить и убрать")),
            ("world", L("World & save", "Мир и сохранение"),
             L("world settings, bodies", "настройки мира, параметры тел")),
            ("language", L("Language", "Язык"), "English · Русский"),
            ("refresh", L("Refresh", "Обновить данные"),
             L("reconnect, re-read the save and blueprint", "переподключиться, перечитать сохранение и чертёж")),
            ("exit", L("Exit", "Выход"), ""),
        ]

    def next_step(self) -> tuple[str, str]:
        """(menu action, one-line hint) — what the player should do now."""
        wd = self.world
        if not K.installed() or self.target is None:
            return "weights", L("Install the AI weights first — press 5, then A.",
                                "Сначала установите веса ИИ — нажмите 5, затем A.")
        if not (wd and wd.live):
            return "refresh", L("Start KSP, press “Start Server” in the kRPC window, then Refresh (8).",
                                "Запустите KSP, нажмите «Start Server» в окне kRPC, затем «Обновить» (8).")
        if self.vessel is None:
            return "design", L("No rocket yet: build one in the VAB, or let me design it (4).",
                               "Ракеты пока нет: соберите её в VAB или дайте мне спроектировать (4).")
        if not getattr(self, "_analysed", False):
            return "analysis", L("Check whether this rocket reaches the target (2).",
                                 "Проверьте, долетит ли ракета до цели (2).")
        return "autopilot", L("All set — start the automatic flight (3).",
                              "Всё готово — запускайте автоматический полёт (3).")

    def main_menu(self) -> None:
        self.offer_weights()
        rec0 = self.next_step()[0]
        sel = next((i for i, it in enumerate(self.menu_items()) if it[0] == rec0), 0)
        while True:
            items = self.menu_items()
            w, _ = T.size()
            rec, hint = self.next_step()
            body = [f"  {T.FLAME}★{T.RESET} {T.WHITE}{L('Next:', 'Дальше:')}{T.RESET} {T.SILVER}{hint}{T.RESET}", ""]
            for i, (key, title, desc) in enumerate(items):
                num = str(i + 1) if key != "exit" else "0"
                extra = ""
                if key == "target":
                    tgt = self.target
                    extra = (f"{T.SKY}{tgt.code}{T.RESET} {T.WHITE}{tgt.title}{T.RESET}" if tgt
                             else f"{T.BAD}{L('locked — install weights', 'закрыто — установите веса')}{T.RESET}")
                if key == rec:
                    title = title + " ★"
                if i == sel:
                    line = (f"  {T.SEL_BG}{T.FLAME} ▸ {T.WHITE}{T.BOLD}{num}  {T.pad(title, 26)}"
                            f"{T.RESET}{T.SEL_BG} {T.pad(T.MUTED + desc, 52)}{T.RESET}")
                else:
                    line = f"    {T.STEEL}{num}  {T.SILVER}{T.pad(title, 26)}{T.RESET} {T.MUTED}{desc}{T.RESET}"
                if extra:
                    line += "   " + extra
                body.append(line)
                if key in ("target", "design", "weights"):
                    body.append("")
            body += [""] + self.status_panel(w)
            if w >= 110:
                body = RK.overlay(body, self._corner_logo(), w - 36)
            self.screen(L("MAIN MENU", "ГЛАВНОЕ МЕНЮ"), body,
                        L("↑↓ — select   Enter — open   digit — jump   Esc — exit",
                          "↑↓ — выбор   Enter — открыть   цифра — быстрый переход   Esc — выход"))
            key = T.read_key(1.0)
            if key is None:
                continue
            if key == T.UP:
                sel = (sel - 1) % len(items)
            elif key == T.DOWN:
                sel = (sel + 1) % len(items)
            elif key == T.ESC:
                return
            elif key == T.ENTER or (isinstance(key, str) and key.isdigit()):
                if key != T.ENTER:
                    idx = int(key) - 1 if key != "0" else len(items) - 1
                    if not 0 <= idx < len(items):
                        continue
                    sel = idx
                action = items[sel][0]
                if action == "exit":
                    return
                getattr(self, f"do_{action}")()
                nxt = self.next_step()[0]
                if nxt != rec:
                    sel = next((i for i, it in enumerate(self.menu_items()) if it[0] == nxt), sel)

    def _corner_logo(self) -> list[str]:
        """Static patch logo for the menu corner — drawn once, then reused."""
        if not hasattr(self, "_logo_cache"):
            try:
                self._logo_cache = LG.emblem(34, 17)
            except Exception:
                self._logo_cache = []
        return self._logo_cache

    def status_panel(self, w: int) -> list[str]:
        wd = self.world
        st = wd.settings
        rows = []
        mode = {"SANDBOX": L("sandbox", "песочница"), "CAREER": L("career", "карьера"),
                "SCIENCE_SANDBOX": L("science", "наука")}.get(st.mode, st.mode)
        rows.append(f"{T.MUTED}{T.pad(L('World', 'Мир'), 11)}{T.RESET}{wd.home} → {wd.moon}"
                    + (f"   {T.WARN}{L('non-stock scale', 'нестандартный масштаб')}{T.RESET}" if wd.scaled else "")
                    + f"   {T.MUTED}{L('mode', 'режим')}{T.RESET} {mode}   {T.MUTED}KSP{T.RESET} {st.version or '—'}")
        if st.commnet:
            comm = L(f"on, range ×{st.range_modifier:g}, DSN lvl {st.tracking_level}",
                     f"вкл, дальность ×{st.range_modifier:g}, DSN ур. {st.tracking_level}")
            if st.require_signal:
                comm += L(", control needs signal", ", управление только при связи")
        else:
            comm = L("off", "выключен")
        rows.append(f"{T.MUTED}{T.pad('CommNet', 11)}{T.RESET}{comm}")
        if self.vessel:
            v = self.vessel
            rows.append(f"{T.MUTED}{T.pad(L('Blueprint', 'Чертёж'), 11)}{T.RESET}{v.name}   "
                        f"{T.MUTED}{craft.describe_source(v)}{T.RESET}")
            rows.append(f"{' ' * 11}{v.mass:.1f} {L('t', 'т')}, {L('parts', 'деталей')} {len(v.parts)}, "
                        f"Δv ({L('vacuum', 'вакуум')}) {fmt(v.dv_vac_total)} {L('m/s', 'м/с')}")
        else:
            rows.append(f"{T.MUTED}{T.pad(L('Blueprint', 'Чертёж'), 11)}{T.RESET}{T.BAD}"
                        f"{L('not found', 'не найден')}{T.RESET}")
        inst = K.installed()
        unlocked = sorted(K.unlocked_targets())
        rows.append(f"{T.MUTED}{T.pad(L('Weights', 'Веса'), 11)}{T.RESET}"
                    + L(f"modules {len(inst)}/{len(K.MODULE_ORDER)}", f"модулей {len(inst)}/{len(K.MODULE_ORDER)}")
                    + "   " + (L(f"{len(unlocked)} targets: ", f"целей {len(unlocked)}: ")
                             + ", ".join(sorted({t.body for t in self.available_targets()})) if unlocked
                               else f"{T.BAD}{L('no targets unlocked', 'цели не открыты')}{T.RESET}"))
        for note in wd.notes:
            rows.append(f"{T.WARN}▲ {note}{T.RESET}")
        return T.box(L("Status", "Состояние"), rows, min(w - 4, 120))

    def do_language(self) -> None:
        self.choose_language()

    # ======================================================================
    # Target
    # ======================================================================
    def do_target(self) -> None:
        section = L("MISSION TARGET", "ЦЕЛЬ МИССИИ")
        if self.no_target(section):
            return
        items = self.available_targets()
        locked = [t for t in ms.targets(self.world) if t not in items and t.code not in {i.code for i in items}]
        sel = next((i for i, t in enumerate(items) if t.code == self.target_code), 0)
        ms_ = L("m/s", "м/с")
        # Budgets are computed once: a planet takes ~0.5 s (Lambert over the window)
        res = self.work(section, [(L("Computing Δv budgets for every target", "Считаю бюджеты Δv для всех целей"),
                                   lambda cb: {t.code: self.budget_of(t) for t in items})])
        if res is None:
            return
        budgets = res[0]
        while True:
            body = [f"  {T.SILVER}{L('Where to?', 'Куда летим?')}{T.RESET}", ""]
            for i, t in enumerate(items):
                b = budgets[t.code]
                alt = L("surface", "поверхность") if t.landing else f"H = {fmt(t.altitude / 1000)} {L('km', 'км')}"
                where = f"{t.body}, {alt}"
                cur = f" {T.OK}✓{T.RESET}" if t.code == self.target_code else "  "
                if i == sel:
                    body.append(f"  {T.SEL_BG}{T.FLAME} ▸ {T.SKY}{T.BOLD}{T.pad(t.code, 9)}{T.RESET}{T.SEL_BG}"
                                f"{T.WHITE}  {T.pad(t.title, 36)} {T.pad(where, 26)}"
                                f" Δv ≈ {T.pad(fmt(b.total), 6, 'right')} {ms_} {T.RESET}{cur}")
                else:
                    body.append(f"    {T.SKY}{T.pad(t.code, 9)}{T.RESET}  {T.SILVER}{T.pad(t.title, 36)}{T.RESET}"
                                f" {T.MUTED}{T.pad(where, 26)} Δv ≈ "
                                f"{T.pad(fmt(b.total), 6, 'right')} {ms_}{T.RESET}{cur}")
            for t in locked:
                body.append(f"    {T.MUTED}{t.code}  {T.pad(t.title, 30)} "
                            f"{L('locked: not in the installed weights', 'закрыто: нет в установленных весах')}{T.RESET}")
            t = items[sel]
            b = budgets[t.code]
            body += ["", f"  {T.MUTED}{L('Budget', 'Бюджет')} «{t.title}»:{T.RESET}"]
            for leg in b.legs:
                body.append(f"    {T.STEEL}•{T.RESET} {T.pad(leg.title, 48)} {T.WHITE}"
                            f"{T.pad(fmt(leg.dv), 6, 'right')} {ms_}{T.RESET}")
            self.screen(section, body, L("↑↓ — select   Enter — set target   Esc — back",
                                         "↑↓ — выбор   Enter — назначить цель   Esc — назад"))
            key = T.read_key(1.0)
            if key == T.UP:
                sel = (sel - 1) % len(items)
            elif key == T.DOWN:
                sel = (sel + 1) % len(items)
            elif key == T.ENTER:
                self.save_target(items[sel].code)
                return
            elif key == T.ESC:
                return

    # ======================================================================
    # Pre-flight analysis
    # ======================================================================
    def analysis_lines(self, rep: analysis.Report) -> list[str]:
        ms_ = L("m/s", "м/с")
        body = []
        t = rep.target
        body.append(f"  {T.SILVER}{L('Target:', 'Цель:')}{T.RESET} {T.SKY}{t.code}{T.RESET} {t.title}"
                    f"   {T.SILVER}{L('Blueprint:', 'Чертёж:')}{T.RESET} {rep.vessel.name if rep.vessel else '—'}")
        body.append("")
        for c in rep.checks:
            body.append("   " + T.status_line(c.status, c.name, c.text, 24))
        body.append("")
        if rep.allocation:
            body.append(f"  {T.MUTED}{L('Δv by stage:', 'Раскладка Δv по ступеням:')}{T.RESET}")
            for leg, used, short in rep.allocation.per_leg:
                stg = ", ".join(str(s) for s in dict.fromkeys(used)) or "—"
                warn = (f"  {T.BAD}{L(f'{fmt(short)} {ms_} short', f'не хватает {fmt(short)} {ms_}')}{T.RESET}"
                        if short > 1 else "")
                body.append(f"    {T.STEEL}•{T.RESET} {T.pad(leg.title, 50)} "
                            f"{T.pad(fmt(leg.dv), 6, 'right')} {ms_}   "
                            f"{T.MUTED}{L('stage', 'ступень')} {stg}{T.RESET}{warn}")
            body.append(f"    {T.MUTED}{T.pad(L('total', 'итого'), 50)} {T.pad(fmt(rep.budget.total), 6, 'right')} "
                        f"{ms_}   {L('with margin', 'с запасом')} {fmt(rep.budget.total_margin)} {ms_}{T.RESET}")
        body.append("")
        verdict = {"ok": (T.OK, L("READY TO FLY: all systems nominal", "ГОТОВ К ПОЛЁТУ: все системы в норме")),
                   "warn": (T.WARN, L("FLIGHT POSSIBLE WITH RISK: see the remarks",
                                      "ВОЗМОЖЕН ПОЛЁТ С РИСКОМ: есть замечания")),
                   "bad": (T.BAD, L("NOT READY: the target is out of reach with this blueprint",
                                    "НЕ ГОТОВ: цель недостижима с этим чертежом"))}[rep.worst]
        body.append(f"  {verdict[0]}{T.BOLD}{verdict[1]}{T.RESET}")
        return body

    def _read_vessel(self, cb) -> craft.Vessel | None:
        if not self.demo:
            self.vessel = craft.loaded(self.world, cb) or self.vessel
        return self.vessel

    def do_analysis(self) -> None:
        section = L("PRE-FLIGHT ANALYSIS", "ПРЕДПОЛЁТНЫЙ АНАЛИЗ")
        if self.no_target(section):
            return
        res = self.work(section, [
            (L("Reading the loaded blueprint", "Считываю загруженный чертёж"), self._read_vessel),
            (L(f"Computing the Δv budget for “{self.target.title}”",
               f"Считаю бюджет Δv под цель «{self.target.title}»"),
             lambda cb: ms.budget(self.world, self.target)),
            (L("Splitting Δv across stages, checking the trained profile",
               "Раскладываю Δv по ступеням и сверяюсь с обученным профилем"),
             lambda cb: analysis.analyse(self.world, self.target, self.vessel)),
        ])
        if res is None:
            return
        self._analysed = True
        self.wait_key(section, self.analysis_lines(res[2]),
                      L("Enter / Esc — back      green — fine · yellow — marginal · red — not enough",
                        "Enter / Esc — назад      зелёный — норма · жёлтый — на грани · красный — не хватит"))

    # ======================================================================
    # Automatic flight
    # ======================================================================
    def ask_yes(self, section: str, body: list[str], question: str) -> bool:
        body = body + ["", f"  {T.WARN}{T.BOLD}{question}{T.RESET}  "
                           f"{T.WHITE}[{L('Y — yes / N — no', 'Y — да / N — нет')}]{T.RESET}"]
        while True:
            self.screen(section, body, L("Y — yes   N / Esc — no", "Y — да   N / Esc — нет"))
            key = T.read_key(1.0)
            if key in ("y", "Y", "н", "Н"):
                return True
            if key in ("n", "N", "т", "Т", T.ESC):
                return False

    def do_autopilot(self) -> None:
        section = L("AUTOMATIC FLIGHT", "АВТОМАТИЧЕСКИЙ ПОЛЁТ")
        if self.no_target(section):
            return
        if not self.demo:
            wd = self.world
            if not wd.live or wd.connection is None or not wd.connection.is_alive():
                self.world = W.reconnect(wd, self.settings)
                wd = self.world
            if not wd.live:
                self.wait_key(section, [
                    f"  {T.BAD}✗ {L('No link to the game.', 'Нет связи с игрой.')}{T.RESET}", "",
                    "  " + L("Start KSP, load the save, open the kRPC window,",
                             "Запустите KSP, загрузите сохранение, откройте окно kRPC"),
                    "  " + L("press “Start Server”, then “Refresh”.",
                             "и нажмите «Start Server», затем «Обновить данные».")])
                return
            if not wd.connection.in_flight():
                if not self.offer_launch(section):
                    return

        box: dict = {}

        def read(cb):
            box["vessel"] = self._read_vessel(cb)
            return box["vessel"]

        def check(cb):
            box["rep"] = analysis.analyse(self.world, self.target, box["vessel"])
            return box["rep"]

        def comms(cb):
            box["need"] = RL.assess(self.world, self.target, box["vessel"], box["rep"].dv_available)
            return box["need"]

        res = self.work(section, [
            (L("Reading the craft on the pad", "Считываю аппарат на стартовом столе"), read),
            (L("Checking it is enough for the target", "Проверяю, хватит ли его на цель"), check),
            (L("Asking the navigator about the link home and relays",
               "Спрашиваю навигатора о связи с домом и ретрансляторах"), comms),
        ])
        if res is None:
            return
        self.vessel = box["vessel"]
        rep = box["rep"]
        target = self.target
        need = box.get("need")
        if need is not None and need.abort:
            if not self.ask_yes(section, self.analysis_lines(rep) + [
                    "", f"  {T.BAD}✗ {L('The navigator advises against this flight:', 'Навигатор не советует лететь:')}{T.RESET}",
                    f"    {need.reason}"], L("Fly anyway?", "Всё равно лететь?")):
                return
        elif need is not None and need.missing > 0:
            info = [f"  {T.WARN}▲ {L('No reliable link home without relays.', 'Без ретрансляторов надёжной связи с домом не будет.')}{T.RESET}",
                    "    " + L(f"Needed: {need.needed} on a {need.orbit_name} orbit of {target.body}, "
                              f"already there: {need.present}.",
                              f"Нужно: {need.needed} на орбите «{need.orbit_name}» у {target.body}, "
                              f"уже стоит: {need.present}.")]
            if RL.is_relay(self.vessel):
                slot = need.present
                if self.ask_yes(section, info, L(f"This craft carries a relay antenna. Deliver it as relay "
                                                 f"{slot + 1} of {need.needed}?",
                                                 f"У этого аппарата есть антенна-ретранслятор. Доставить его как "
                                                 f"ретранслятор {slot + 1} из {need.needed}?")):
                    target = RL.relay_target(self.world, target, slot, need.needed, need.orbit_class)
                elif not self.ask_yes(section, info, L("Fly the main mission without a reliable link?",
                                                        "Лететь основной миссией без надёжной связи?")):
                    return
            else:
                info += ["", "    " + L("Build a relay satellite (Rocket design → Mission: relay), put it on "
                                       "the pad and start the automatic flight — it will be delivered first.",
                                       "Постройте спутник-ретранслятор (Проектирование → Задача: ретранслятор), "
                                       "выведите на стол и запустите автополёт — он будет доставлен первым.")]
                if not self.ask_yes(section, info, L("Fly the main mission without a reliable link?",
                                                     "Лететь основной миссией без надёжной связи?")):
                    return

        res = self.work(section, [
            (L("Locating the planets; planning launch window, heading, phase angle",
               "Смотрю, где сейчас планеты; строю план: окно пуска, азимут, фазовый угол"),
             lambda cb: autopilot.build(self.world, target, box["vessel"], rep)),
        ])
        if res is None:
            return
        box["plan"] = res[0]
        if rep.worst != "ok":
            yellow = sum(1 for c in rep.checks if c.status == "warn")
            red = sum(1 for c in rep.checks if c.status == "bad")
            q = L(f"Remarks: {yellow} yellow, {red} red. Fly anyway?",
                  f"Замечаний: жёлтых {yellow}, красных {red}. Точно лететь?")
            if not self.ask_yes(section, self.analysis_lines(rep), q):
                return
        if not self.preview(box["plan"]):
            return
        self.fly(box["plan"], rep)

    def offer_launch(self, section: str) -> bool:
        path = craft.latest_craft(self.world.ksp_root, self.world.save_name)
        if path is None:
            self.wait_key(section, [f"  {T.BAD}✗ {L('The game is not in flight and the save has no blueprints.', 'Игра не в полёте, а чертежей в сохранении нет.')}{T.RESET}"])
            return False
        folder = path.parent.name
        ok = self.ask_yes(section, [
            "  " + L(f"The game is in the “{self.world.connection.game_scene()}” scene.",
                     f"Игра сейчас в сцене «{self.world.connection.game_scene()}»."),
            "  " + L("Latest blueprint:", "Последний чертёж:") + f" {T.WHITE}{path.stem}{T.RESET} ({folder})"],
            L("Put it on the launch pad?", "Вывести его на стартовый стол?"))
        if not ok:
            return False
        try:
            site = "LaunchPad" if folder == "VAB" else "Runway"
            self.world.connection.space_center.launch_vessel(folder, path.stem, site)
            time.sleep(6.0)
            self.world = W.reconnect(self.world, self.settings)
            return self.world.connection is not None and self.world.connection.in_flight()
        except Exception as exc:
            self.wait_key(section, [f"  {T.BAD}✗ {L('Could not launch the craft:', 'Не удалось вывести аппарат:')} {exc}{T.RESET}"])
            return False

    def preview(self, plan: autopilot.FlightPlan) -> bool:
        name = self.vessel.name if self.vessel else "—"
        while True:
            plan.set_baseline(time.time())
            lines = autopilot.render(plan, name, footer=L("Enter — fly the plan   Esc — cancel",
                                                          "Enter — старт по плану   Esc — отмена"))
            lines.insert(3, f" {T.WHITE}{T.BOLD}{L('FLIGHT PLAN', 'ПЛАН ПОЛЁТА')}{T.RESET} "
                            f"{T.MUTED}{L('— computed, waiting for confirmation', '— рассчитан, ожидает подтверждения')}{T.RESET}")
            T.draw(lines)
            key = T.read_key(1.0)
            if key == T.ENTER:
                return True
            if key == T.ESC:
                return False

    def fly(self, plan: autopilot.FlightPlan, rep) -> None:
        name = self.vessel.name if self.vessel else "—"
        if self.demo:
            autopilot.demo_run(plan, speed=4.0)
            ex = None
        else:
            ex = autopilot.Executor(self.world, plan, self.vessel, rep)
            ex.start()
            while not plan.started and not plan.finished:
                time.sleep(0.1)
        confirm = False
        while True:
            footer = (L("Esc — abort the flight", "Esc — прервать полёт") if not plan.finished
                      else L("Enter / Esc — back to menu", "Enter / Esc — вернуться в меню"))
            if confirm:
                footer = L("Abort? Y — yes, throttle to zero, stop after this step   N — continue",
                           "Прервать полёт? Y — да, газ в ноль и остановка после текущего шага   N — продолжить")
            if plan.question and not plan.finished:
                left = max(0, int(plan.decide_by - time.time()))
                footer = (f"{T.WARN}▲ {plan.question}{T.RESET}  " + L(
                    f"R — retry   M — take manual control   S — rescue the craft   (rescue in {left} s)",
                    f"R — повторить   M — взять управление   S — спасти аппарат   (спасение через {left} с)"))
            tele = ex.telemetry_line() if ex else ""
            # The flight code writes its journal in Russian — shown in Russian mode only
            logs = ex.tail.lines if ex and i18n.ru() else None
            if ex and ex.error and plan.finished:
                plan.verdict = plan.verdict or ex.error
            T.draw(autopilot.render(plan, name, tele, logs, footer))
            key = T.read_key(0.25)
            if plan.finished and key in (T.ENTER, T.ESC):
                return
            if plan.question and ex and not plan.finished:
                pick = {"r": "retry", "к": "retry", "m": "manual", "ь": "manual",
                        "s": "rescue", "ы": "rescue"}.get(str(key).lower())
                if pick:
                    ex.decide(pick)
                continue
            if confirm:
                if key in ("y", "Y", "н", "Н"):
                    if ex:
                        ex.stop()
                    confirm = False
                elif key in ("n", "N", "т", "Т", T.ESC):
                    confirm = False
            elif key == T.ESC and not plan.finished:
                confirm = True

    # ======================================================================
    # Rocket design
    # ======================================================================
    def do_design(self) -> None:
        section = L("ROCKET DESIGN", "ПРОЕКТИРОВАНИЕ РАКЕТЫ")
        if self.no_target(section):
            return
        cfg = self.design_cfg
        rows = ["relay", "stages", "boosters", "crew", "payload", "nuclear", "go"]
        auto_dv = ms.budget(self.world, self.target, gear=True).total_margin   # once, not per redraw
        sel = 0
        while True:
            def val(key):
                if key == "relay":
                    return (L("relay satellite for this target", "спутник-ретранслятор для этой цели")
                            if cfg.relay else L("main craft", "основной аппарат"))
                if key == "stages":
                    auto = cfg.stage_count(self.target, auto_dv)
                    return (L(f"auto ({auto})", f"авто ({auto})")
                            if not cfg.stages else str(cfg.stages))
                if key == "boosters":
                    return L("none", "нет") if not cfg.boosters else L(f"{cfg.boosters} solid", f"{cfg.boosters} твердотопливных")
                if key == "crew":
                    return {0: L("probe (no crew)", "зонд (без экипажа)"), 1: L("1 kerbal", "1 кербонавт"),
                            3: L("3 kerbals", "3 кербонавта")}[cfg.crew]
                if key == "payload":
                    return f"{cfg.payload:.2f} {L('t', 'т')}"
                if key == "nuclear":
                    return L("allowed on vacuum stages", "можно на вакуумных ступенях") if cfg.nuclear else L("no", "нет")
                return ""
            names = {"relay": L("Mission", "Задача"), "stages": L("Stages", "Ступеней"), "boosters": L("Side boosters", "Боковые ускорители"),
                     "crew": L("Crew", "Экипаж"), "payload": L("Extra payload", "Доп. нагрузка"),
                     "nuclear": L("Nuclear engine LV-N", "Ядерный двигатель LV-N"),
                     "go": L("▶ Design the rocket", "▶ Спроектировать ракету")}
            body = [f"  {T.SILVER}{L('Target:', 'Цель:')}{T.RESET} {T.SKY}{self.target.code}{T.RESET} {self.target.title}",
                    f"  {T.MUTED}{L('Set up the rocket first — every layout gives a different rocket.', 'Сначала настройте ракету — каждая схема даёт другую ракету.')}{T.RESET}", ""]
            for i, key in enumerate(rows):
                if key == "go":
                    body.append("")
                label = f"{T.pad(names[key], 28)} {T.WHITE}{val(key)}{T.RESET}"
                if key != "go":
                    label = f"{T.pad(names[key], 28)} {T.MUTED}◂{T.RESET} {T.WHITE}{val(key)}{T.RESET} {T.MUTED}▸{T.RESET}"
                if i == sel:
                    body.append(f"  {T.SEL_BG}{T.FLAME} ▸ {T.RESET}{T.SEL_BG}{T.BOLD}{label}{T.RESET}")
                else:
                    body.append(f"    {label}")
            self.screen(section, body, L("↑↓ — row   ←→ — change   Enter — design   Esc — back",
                                         "↑↓ — строка   ←→ — изменить   Enter — спроектировать   Esc — назад"))
            key = T.read_key(1.0)
            row = rows[sel]
            step = 1 if key == T.RIGHT else -1 if key == T.LEFT else 0
            if key == T.UP:
                sel = (sel - 1) % len(rows)
            elif key == T.DOWN:
                sel = (sel + 1) % len(rows)
            elif step:
                if row == "relay":
                    cfg.relay = not cfg.relay
                elif row == "stages":
                    cfg.stages = (cfg.stages + step) % 5
                elif row == "boosters":
                    opts = [0, 2, 4, 6, 8]
                    cfg.boosters = opts[(opts.index(cfg.boosters) + step) % len(opts)]
                elif row == "crew":
                    opts = [0, 1, 3]
                    cfg.crew = opts[(opts.index(cfg.crew) + step) % len(opts)]
                elif row == "payload":
                    cfg.payload = max(0.0, round(cfg.payload + 0.25 * step, 2))
                elif row == "nuclear":
                    cfg.nuclear = not cfg.nuclear
            elif key == T.ENTER:
                self.settings["payload_t"] = cfg.payload
                self.save_settings()
                res = self.work(section, [
                    (L("Picking engines and tanks from the game catalog",
                       "Подбираю двигатели и баки из каталога игры"),
                     lambda cb: designer.design(self.world, self.target, cfg, margin=True)),
                ])
                if res:
                    self.wait_key(section, self.design_lines(res[0]),
                                  L("↑↓ — scroll   Enter / Esc — back to the settings",
                                    "↑↓ — прокрутка   Enter / Esc — назад к настройкам"))
            elif key == T.ESC:
                return

    def design_lines(self, d: designer.Design) -> list[str]:
        ms_, t_, kn = L("m/s", "м/с"), L("t", "т"), L("kN", "кН")
        body = [f"  {T.SILVER}{L('Target:', 'Цель:')}{T.RESET} {T.SKY}{d.target.code}{T.RESET} {d.target.title}"
                f"   {T.SILVER}{L('Total mass:', 'Полная масса:')}{T.RESET} {T.WHITE}{d.total_mass:.1f} {t_}{T.RESET}"
                f"   {T.SILVER}Δv:{T.RESET} {T.WHITE}{fmt(d.total_dv)}{T.RESET} / {fmt(d.budget.total_margin)} {ms_}"
                f" {T.MUTED}({L('need with 15 % margin', 'нужно с запасом 15 %')}){T.RESET}", ""]
        body.append(f"  {T.MUTED}{T.pad(L('Maneuver', 'Манёвр'), 50)}{T.pad(L('exact', 'впритык'), 12, 'right')}"
                    f"{T.pad(L('with margin', 'с запасом'), 14, 'right')}{T.RESET}")
        for leg in d.budget.legs:
            body.append(f"   {T.pad(leg.title, 50)}{T.pad(fmt(leg.dv) + ' ' + ms_, 12, 'right')}"
                        f"{T.OK}{T.pad(fmt(leg.dv * ms.MARGIN) + ' ' + ms_, 14, 'right')}{T.RESET}")
        body.append("")
        # Stages from the top of the rocket down to the launch pad — how it is built in the VAB
        for st in reversed(d.stages):
            head = (f"  {T.FLAME}{T.BOLD}{st.label}{T.RESET} {T.MUTED}— {st.role}{T.RESET}")
            if not st.feasible or st.engine is None:
                body += [head, f"     {T.BAD}✗ {st.note or L('could not be designed', 'не удалось подобрать')}{T.RESET}", ""]
                continue
            body.append(head + f"   {T.WHITE}Δv {fmt(st.dv)} {ms_}{T.RESET}   TWR {st.twr:.2f}"
                        f"   {L('mass', 'масса')} {st.m0:.1f} {t_}   {L('thrust', 'тяга')} {fmt(st.thrust)} {kn}"
                        f"   Isp {st.isp:.0f}")
            body.append(f"     {T.MUTED}{T.pad(L('Engine', 'Двигатель'), 16)}{T.RESET} "
                        f"{T.WHITE}{st.engine.title}{T.RESET} × {st.engine.count}")
            if st.tanks:
                body.append(f"     {T.MUTED}{T.pad(L('Fuel tanks', 'Топливные баки'), 16)}{T.RESET} "
                            + ",  ".join(f"{T.WHITE}{i.title}{T.RESET} × {i.count}" for i in st.tanks))
            body.append(f"     {T.MUTED}{T.pad(L('Fuel', 'Топливо'), 16)}{T.RESET} {st.fuel_type}: "
                        f"{fmt(st.units)} {L('u', 'ед.')} ({st.propellant:.2f} {t_})")
            for i in st.extras:
                body.append(f"     {T.MUTED}{T.pad(L('Also', 'Ещё'), 16)}{T.RESET} {i.title} × {i.count}")
            if st.note:
                body.append(f"     {T.WARN}▲ {st.note}{T.RESET}")
            body.append("")
        body.append(f"  {T.OK}{T.BOLD}{L('REQUIRED', 'ОБЯЗАТЕЛЬНО')}{T.RESET}")
        for kind, item, why in d.required:
            body.append(f"   {T.OK}●{T.RESET} {T.pad(kind, 20)} {T.WHITE}{item.title}{T.RESET} × {item.count}"
                        + (f"   {T.MUTED}{why}{T.RESET}" if why else ""))
        body += ["", f"  {T.SKY}{T.BOLD}{L('OPTIONAL', 'ПО ЖЕЛАНИЮ')}{T.RESET}"]
        for kind, item, why in d.optional:
            body.append(f"   {T.SKY}○{T.RESET} {T.pad(kind, 20)} {T.WHITE}{item.title}{T.RESET} × {item.count}"
                        + (f"   {T.MUTED}{why}{T.RESET}" if why else ""))
        body += ["", f"  {T.MUTED}{L('Masses and Δv are computed from the real parts above. TWR targets come from the trained designer profile.', 'Массы и Δv посчитаны по реальным деталям выше. Требования к TWR — из обученного профиля конструктора.')}{T.RESET}"]
        return body

    # ======================================================================
    # Weights / knowledge
    # ======================================================================
    def offer_weights(self) -> None:
        """First start without weights: find the downloaded archive and set it up."""
        if K.installed() or self.settings.get("weights_offer_declined"):
            return
        section = L("AI WEIGHTS", "ВЕСА ИИ")
        zips = K.find_archives()
        lib = K.scan_library(self.settings)
        have_lib = any(lib.values())
        if not zips and not have_lib:
            return
        src = zips[0] if zips else K.library_root(self.settings)
        if not self.ask_yes(section, [
                f"  {T.WHITE}{L('ASTRAUTOMA has no AI installed yet — it cannot fly without it.', 'В ASTRAUTOMA ещё не установлен ИИ — без него он не летает.')}{T.RESET}", "",
                "  " + L("Found:", "Нашёл:") + f" {T.SKY}{src}{T.RESET}"],
                L("Install it now?", "Установить сейчас?")):
            self.settings["weights_offer_declined"] = True
            self.save_settings()
            return
        self._install_everything(section, zips[0] if zips else None)

    def _install_everything(self, section: str, archive) -> str:
        steps = []
        if archive is not None:
            steps.append((L(f"Unpacking {archive.name}", f"Распаковываю {archive.name}"),
                          lambda cb: K.import_archive(archive, self.settings)))
        steps.append((L("Checking and installing every module", "Проверяю и устанавливаю все модули"),
                      lambda cb: K.install_all(self.settings)))
        res = self.work(section, steps)
        if res is None:
            return f"{T.BAD}✗ {L('Not installed', 'Не установлено')}{T.RESET}"
        self.detect_targets()
        n = len(self.available_targets())
        return f"{T.OK}✓ {L(f'AI installed — {n} targets open', f'ИИ установлен — открыто целей: {n}')}{T.RESET}"

    def detect_targets(self) -> None:
        self._budgets = {}
        threading.Thread(target=self._prefill_budgets, daemon=True).start()

    def do_weights(self) -> None:
        section = L("WEIGHTS / KNOWLEDGE", "ВЕСА / ЗНАНИЯ")
        sel, ver = 0, 0
        message = ""
        while True:
            lib = K.scan_library(self.settings)
            inst = K.installed()
            mod = K.MODULE_ORDER[sel]
            versions = lib.get(mod, [])
            ver = min(ver, max(0, len(versions) - 1))
            root = K.library_root(self.settings)
            body = [f"  {T.MUTED}{L('Library:', 'Библиотека:')}{T.RESET} {root}"
                    + ("" if root.exists() else f"  {T.BAD}{L('(folder not found)', '(папка не найдена)')}{T.RESET}"),
                    f"  {T.MUTED}{L('The app ships without weights; each module unlocks its skills.', 'Приложение идёт без весов; каждый модуль открывает свои умения.')}{T.RESET}", ""]
            for i, m in enumerate(K.MODULE_ORDER):
                have = inst.get(m)
                state = (f"{T.OK}✓ {have.name}{T.RESET}" if have
                         else f"{T.MUTED}{L('not installed', 'не установлен')}{T.RESET}")
                count = len(lib.get(m, []))
                line = (f"{T.pad(m, 15)} {T.pad(K.module_title(m), 28)} {T.pad(state, 34)} "
                        f"{T.MUTED}{L(f'{count} in library', f'в библиотеке: {count}')}{T.RESET}")
                body.append((f"  {T.SEL_BG}{T.FLAME} ▸ {T.RESET}{T.SEL_BG}{line}{T.RESET}") if i == sel else f"    {line}")
            body += ["", f"  {T.SILVER}{mod}{T.RESET} — {K.module_purpose(mod)}"]
            if versions:
                for j, v in enumerate(versions):
                    mark = "▸" if j == ver else " "
                    err = f"  {T.BAD}✗ {v.error}{T.RESET}" if v.error else ""
                    installed_mark = f" {T.OK}({L('installed', 'установлен')}){T.RESET}" if inst.get(mod) and inst[mod].name == v.name else ""
                    body.append(f"    {T.FLAME}{mark}{T.RESET} {T.WHITE}{v.name}{T.RESET}{installed_mark}{err}")
                    if j == ver and v.summary():
                        body.append(f"        {T.MUTED}{v.summary()}{T.RESET}")
            else:
                body.append(f"    {T.MUTED}{L('No versions of this module in the library.', 'В библиотеке нет версий этого модуля.')}{T.RESET}")
            if message:
                body += ["", "  " + message]
            self.screen(section, body, L(
                "A — install all (newest)   I — import a downloaded .zip   ↑↓ — module   ←→ — version   "
                "Enter — install   Del — remove   O — folder   Esc — back",
                "A — установить всё (новейшее)   I — импорт скачанного .zip   ↑↓ — модуль   ←→ — версия   "
                "Enter — установить   Del — убрать   O — папка   Esc — назад"))
            key = T.read_key(1.0)
            if key == T.ESC:
                return
            if key == T.UP:
                sel, ver = (sel - 1) % len(K.MODULE_ORDER), 0
            elif key == T.DOWN:
                sel, ver = (sel + 1) % len(K.MODULE_ORDER), 0
            elif key == T.LEFT and versions:
                ver = (ver - 1) % len(versions)
            elif key == T.RIGHT and versions:
                ver = (ver + 1) % len(versions)
            elif key == T.ENTER and versions:
                try:
                    K.install(versions[ver])
                    message = f"{T.OK}✓ {L('Installed', 'Установлено')}: {mod} · {versions[ver].name}{T.RESET}"
                except Exception as exc:
                    message = f"{T.BAD}✗ {L('Not installed', 'Не установлено')}: {exc}{T.RESET}"
            elif key in ("r", "R", "к", "К", T.DEL):
                if mod in inst:
                    K.remove(mod)
                    message = f"{T.WARN}{L('Removed', 'Убрано')}: {mod}{T.RESET}"
            elif key in ("a", "A", "ф", "Ф"):
                message = self._install_everything(section, None)
            elif key in ("i", "I", "ш", "Ш"):
                zips = K.find_archives()
                if not zips:
                    message = f"{T.WARN}{L('No ASTRAUTOMA-Weights*.zip in Downloads or on the Desktop.', 'Нет ASTRAUTOMA-Weights*.zip в «Загрузках» или на рабочем столе.')}{T.RESET}"
                else:
                    message = self._install_everything(section, zips[0])
            elif key in ("o", "O", "щ", "Щ"):
                try:
                    root.mkdir(parents=True, exist_ok=True)
                    os.startfile(str(root))
                except Exception as exc:
                    message = f"{T.BAD}✗ {exc}{T.RESET}"

    # ======================================================================
    # World
    # ======================================================================
    def do_world(self) -> None:
        wd = self.world
        st = wd.settings
        yes, no = L("yes", "да"), L("no", "нет")
        body = [f"  {T.SILVER}{T.pad(L('KSP install', 'Установка KSP'), 16)}{T.RESET}{wd.ksp_root or '—'}",
                f"  {T.SILVER}{T.pad(L('Save', 'Сохранение'), 16)}{T.RESET}{wd.save_name or '—'}"
                + (f"  {T.MUTED}({L('written', 'записано')} {datetime.fromtimestamp(wd.save_time):%d.%m.%Y %H:%M}){T.RESET}"
                   if wd.save_time else ""),
                f"  {T.SILVER}{T.pad(L('Bodies from', 'Источник тел'), 16)}{T.RESET}"
                + (L("live game (kRPC) — mods and scale included", "живая игра (kRPC) — учтены моды и масштаб")
                   if wd.live else L("stock KSP 1.12 system (game not connected)",
                                     "стандартная система KSP 1.12 (игра не подключена)")),
                "", f"  {T.MUTED}{L('Difficulty settings that affect the flight:', 'Настройки сложности, влияющие на полёт:')}{T.RESET}"]
        rows = [(L("Game mode", "Режим игры"), st.mode), (L("Version", "Версия"), st.version or "—"),
                (L("Mods", "Моды"), yes if st.modded else no),
                ("CommNet", L("on", "включён") if st.commnet else L("off", "выключен")),
                (L("Control without signal", "Управление без связи"),
                 L("forbidden", "запрещено") if st.require_signal else L("allowed", "разрешено")),
                (L("Plasma blackout", "Потеря связи в плазме"), yes if st.plasma_blackout else no),
                (L("Range modifier", "Множитель дальности"), f"×{st.range_modifier:g}"),
                (L("DSN modifier", "Множитель DSN"), f"×{st.dsn_modifier:g}"),
                (L("Tracking station", "Станция слежения"), L(f"level {st.tracking_level}", f"уровень {st.tracking_level}")),
                (L("Ground stations", "Наземные станции"), yes if st.ground_stations else no),
                (L("Reentry heating", "Нагрев при входе"), f"×{st.reentry_heat:g}")]
        for k, v in rows:
            body.append(f"    {T.pad(k, 28)} {T.WHITE}{v}{T.RESET}")
        body += ["", f"  {T.MUTED}{L('Bodies:', 'Тела:')}{T.RESET}"]
        km = L("km", "км")
        for name in (wd.home, wd.moon):
            b = wd.bodies[name]
            line = (f"    {T.SKY}{T.pad(name, 10)}{T.RESET} R = {b.radius / 1000:,.0f} {km}   "
                    f"g = {b.g0:.2f} {L('m/s²', 'м/с²')}   {L('day', 'сутки')} {b.rotation_period / 3600:.2f} {L('h', 'ч')}   "
                    f"{L('atmosphere', 'атмосфера')} {b.atmosphere_depth / 1000:.0f} {km}")
            if name == wd.moon:
                line += f"   {L('orbit', 'орбита')} {b.sma / 1e6:,.1f} {L('Mm', 'тыс. км')}"
            body.append(line.replace(",", " "))
        self.wait_key(L("WORLD & SAVE", "МИР И СОХРАНЕНИЕ"), body)

    def do_refresh(self) -> None:
        self.screen(L("REFRESH", "ОБНОВЛЕНИЕ"),
                    [f"  {T.MUTED}{L('Reconnecting and reading the save…', 'Переподключение и чтение сохранения…')}{T.RESET}"], "")
        if self.world and self.world.connection:
            try:
                self.world.connection.close()
            except Exception:
                pass
        self.detect()

    # ======================================================================
    def run(self) -> int:
        T.enable_vt()
        T.alt_screen(True)
        T.hide_cursor()
        try:
            T.clear()
            if "language" not in self.settings:
                self.choose_language(first=True)
            T.clear()
            self.splash()
            T.clear()
            self.main_menu()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            T.show_cursor()
            T.alt_screen(False)
            if self.world and self.world.connection:
                try:
                    self.world.connection.close()
                except Exception:
                    pass
        return 0
