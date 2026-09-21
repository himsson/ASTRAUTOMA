"""Подготовка судна к попытке: чертёж -> VAB -> стартовый стол.

Стратегии (по убыванию предпочтения):
1. generate  — сгенерировать .craft из проекта и запустить его;
2. existing  — запустить уже имеющийся в VAB чертёж с тем же именем;
3. revert    — откатиться к старту текущего судна;
4. as_is     — работать с тем судном, что уже на столе.

Отдельная важная деталь: KSP ищет чертёж в каталоге ТЕКУЩЕГО сохранения,
а его имя ниоткуда не следует. Мы определяем его надёжно — просим игру
сделать быстрое сохранение с меткой и смотрим, в какой папке она появилась.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..config import CONFIG
from ..engineer.craft_writer import Blueprint, CraftAssembler
from ..logging_setup import get_logger

log = get_logger("agent.provision")

KSP_ROOT_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Steam\steamapps\common\Kerbal Space Program"),
    Path(r"C:\Program Files\Steam\steamapps\common\Kerbal Space Program"),
    Path(r"C:\Games\Kerbal Space Program"),
    Path(r"D:\Steam\steamapps\common\Kerbal Space Program"),
    Path(r"D:\SteamLibrary\steamapps\common\Kerbal Space Program"),
    Path(r"D:\Games\Kerbal Space Program"),
    Path(r"E:\SteamLibrary\steamapps\common\Kerbal Space Program"),
    Path.home() / "Kerbal Space Program",
]

MARKER_NAME = "kia_save_probe"


def find_ksp_root() -> Path | None:
    """Ищет каталог установки KSP по типовым путям Windows/Steam."""
    if CONFIG.craft.ksp_root:
        path = Path(CONFIG.craft.ksp_root)
        if path.exists():
            return path
    for path in KSP_ROOT_CANDIDATES:
        if (path / "KSP_x64.exe").exists() or (path / "KSP.exe").exists():
            return path
    return None


@dataclass
class ProvisionResult:
    ok: bool
    strategy: str
    craft_path: str | None = None
    message: str = ""


class VesselProvisioner:
    """Выводит на стартовый стол судно, соответствующее проекту."""

    def __init__(self, connection, control=None, craft_config=None):
        self.connection = connection
        self.sc = connection.space_center
        self.control = control
        self.cfg = craft_config or CONFIG.craft
        self._save_detected = False

    # ------------------------------------------------------------------
    def detect_ksp_root(self) -> str | None:
        if self.cfg.ksp_root:
            return self.cfg.ksp_root
        path = find_ksp_root()
        if path is not None:
            log.info("Найден каталог KSP: %s", path)
            self.cfg.ksp_root = str(path)
            return str(path)
        log.warning("Каталог KSP не найден — .craft писать некуда")
        return None

    def detect_save_name(self, force: bool = False) -> str:
        """Определяет имя ЗАГРУЖЕННОГО сохранения через метку-quicksave."""
        if self._save_detected and not force:
            return self.cfg.save_name
        root = self.detect_ksp_root()
        if not root:
            return self.cfg.save_name
        saves = Path(root) / "saves"
        if not saves.exists():
            return self.cfg.save_name
        try:
            self.sc.save(MARKER_NAME)
            time.sleep(1.5)
            hits = list(saves.glob(f"*/{MARKER_NAME}.sfs"))
            if hits:
                name = hits[0].parent.name
                if name != self.cfg.save_name:
                    log.info("Активное сохранение KSP: '%s' (было '%s')",
                             name, self.cfg.save_name)
                self.cfg.save_name = name
                self._save_detected = True
            for hit in hits:
                try:
                    hit.unlink()
                except OSError:
                    pass
        except Exception as exc:
            log.debug("Метку сохранения поставить не удалось: %s", exc)
        return self.cfg.save_name

    # ------------------------------------------------------------------
    def craft_directory(self) -> Path | None:
        root = self.detect_ksp_root()
        if not root:
            return None
        self.detect_save_name()
        return Path(root) / "saves" / self.cfg.save_name / "Ships" / "VAB"

    def write_craft(self, blueprint: Blueprint) -> Path | None:
        directory = self.craft_directory()
        if directory is None:
            return None
        assembler = CraftAssembler(blueprint.name or self.cfg.craft_name)
        return assembler.save(blueprint, directory)

    # ------------------------------------------------------------------
    def provision(self, blueprint: Blueprint | None) -> ProvisionResult:
        """Готовит судно на столе под указанный чертёж."""
        control = self.control
        if control is None:
            from ..environment.control import ShipControl
            control = ShipControl(self.connection)
        craft_name = (blueprint.name if blueprint else None) or self.cfg.craft_name

        # ЗАКРЕПЛЁННЫЙ ЧЕРТЁЖ ПЕРЕБИВАЕТ ВСЁ. Оператор велел летать на
        # одной конкретной ракете и отключить собственные — значит ни
        # писать свой чертёж, ни поднимать его со стола нельзя.
        fixed = getattr(self.cfg, "fixed_craft", "")
        if fixed:
            if control.launch_craft(fixed) and self.wait_for_vessel():
                return ProvisionResult(True, "fixed", None,
                                       f"Закреплённый чертёж «{fixed}» "
                                       f"выведен на стартовый стол")
            log.error("Закреплённый чертёж «%s» не запустился", fixed)
            if control.revert_to_launch() and self.wait_for_vessel():
                return ProvisionResult(True, "revert", None,
                                       "Откат к моменту старта")
            return ProvisionResult(False, "fixed", None,
                                   f"Чертёж «{fixed}» не найден в ангаре")

        if blueprint is not None and self.cfg.generate_craft_files:
            path = self.write_craft(blueprint)
            if path is not None and self.cfg.launch_from_vab:
                if control.launch_craft(craft_name) and self.wait_for_vessel():
                    return ProvisionResult(True, "generate", str(path),
                                           f"Сгенерированный чертёж '{craft_name}' "
                                           f"выведен на стартовый стол")
                log.warning("KSP не принял сгенерированный чертёж — запасные пути")

        if self.cfg.launch_from_vab and control.launch_craft(craft_name):
            if self.wait_for_vessel():
                return ProvisionResult(True, "existing", None,
                                       "Запущен существующий чертёж из VAB")

        if control.revert_to_launch() and self.wait_for_vessel():
            return ProvisionResult(True, "revert", None, "Откат к моменту старта")

        try:
            vessel = self.sc.active_vessel
            situation = str(vessel.situation).split(".")[-1]
            if situation in ("pre_launch", "landed", "splashed"):
                return ProvisionResult(True, "as_is", None,
                                       f"Используем судно '{vessel.name}' на столе")
            return ProvisionResult(False, "as_is", None,
                                   f"Судно в состоянии '{situation}' — старт невозможен")
        except Exception as exc:
            return ProvisionResult(False, "none", None, f"Нет активного судна: {exc}")

    # ------------------------------------------------------------------
    def wait_for_vessel(self, timeout: float = 90.0) -> bool:
        """Ждёт, пока судно окажется в полётной сцене и станет доступным."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.connection.in_flight():
                    vessel = self.sc.active_vessel
                    if vessel is not None and len(vessel.parts.all) > 0:
                        return True
            except Exception:
                pass
            time.sleep(1.5)
        log.warning("Судно не появилось в полётной сцене за %.0f с", timeout)
        return False

    # Совместимость со старым именем
    def wait_for_scene(self, timeout: float = 60.0) -> bool:
        return self.wait_for_vessel(timeout)
