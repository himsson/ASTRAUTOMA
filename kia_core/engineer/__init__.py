"""Модуль Инженера: каталог деталей игры, расчёты, автономное проектирование."""
from . import parts_db, rocket_math
from .autodesign import (AutonomousEngineer, DeltaVBudget, DesignPolicy,
                         StageRequirement, StageSolution, VehicleDesign,
                         compute_budget, plan_stages)
from .craft_writer import Blueprint, CraftAssembler, StageBuild
from .design_report import render_markdown, write_report
from .parts_catalog import PartCatalog, PartInfo, get_catalog

__all__ = ["AutonomousEngineer", "DesignPolicy", "VehicleDesign", "DeltaVBudget",
           "StageRequirement", "StageSolution", "compute_budget", "plan_stages",
           "CraftAssembler", "Blueprint", "StageBuild",
           "PartCatalog", "PartInfo", "get_catalog",
           "write_report", "render_markdown", "parts_db", "rocket_math"]
