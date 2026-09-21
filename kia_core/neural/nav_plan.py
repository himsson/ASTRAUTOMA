"""Навигатор — только применение готовых весов (без обучения).

Этот файл одинаков в папке обучения и в ASTRAUTOMA: сеть навигатора,
загрузка весов (безопасно, weights_only=True) и план миссии для игры.
Обучение — в navigator.py, которого в ASTRAUTOMA нет.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..config import ROOT

MODEL_FILE = ROOT / "kia_model_navigator.pth"
BEST_FILE = ROOT / "kia_model_navigator_best.pth"


# ==========================================================================
def build_agent():
    """Сеть навигатора. Маленькая, считается на процессоре."""
    from .nav_env import CONTINUOUS_DIM, DISCRETE_SIZES, OBS_DIM
    from .ppo import PPOAgent, PPOConfig
    obs_dim = OBS_DIM
    cfg = PPOConfig(learning_rate=2e-5, gamma=0.0, gae_lambda=0.0, epochs=6,
                    minibatch_size=512, entropy_coef=0.0005, target_kl=0.02,
                    anneal_lr=False)
    agent = PPOAgent(obs_dim=obs_dim, continuous_dim=CONTINUOUS_DIM,
                     discrete_sizes=DISCRETE_SIZES, config=cfg, hidden=256,
                     depth=3, name="навигатор")
    # Процессор: сеть маленькая, пачки большие — пересылка на видеокарту
    # съела бы весь выигрыш
    agent.device = torch.device("cpu")
    agent.net.to(agent.device)
    agent.optimizer = torch.optim.Adam(agent.net.parameters(), lr=cfg.learning_rate, eps=1e-5)
    return agent


def save(agent, stage: int, path: Path = MODEL_FILE, extra: dict | None = None) -> Path:
    payload = {"navigator": agent.state_dict(), "stage": int(stage),
               "format": "kia-navigator-1", "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load(agent, path: Path = MODEL_FILE) -> int | None:
    """Грузит веса. Возвращает ступень или None."""
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if agent.load_state_dict(payload["navigator"]):
        return int(payload.get("stage", 1))
    return None


# ==========================================================================
# Применение: план миссии для игры
# ==========================================================================
def plan_mission(target: str, objective: str, dv: float, *, ut: float = 0.0,
                 heatshield: bool = True, chutes: bool = True, antenna: float = 5e3,
                 commnet: bool = True, require: bool = False, range_mod: float = 1.0,
                 dsn: int = 3, relays_there: int = 0, path: Path = MODEL_FILE) -> dict:
    """План от обученного навигатора для реальной обстановки."""
    from .nav_env import NavigatorEnv, OBJECTIVES, RELAY_CLASSES
    from .solar import PLANETS
    agent = build_agent()
    if load(agent, path) is None:
        raise FileNotFoundError(f"Нет обученных весов навигатора: {path}")
    env = NavigatorEnv(batch=1, stage=3)
    n = 1
    env.ctx = {"target": np.array([PLANETS.index(target)]),
               "objective": np.array([OBJECTIVES.index(objective)]),
               "ut0": np.array([ut]), "heatshield": np.array([heatshield]),
               "chutes": np.array([chutes]), "commnet": np.array([commnet]),
               "require": np.array([require]), "range_mod": np.array([range_mod]),
               "dsn": np.array([dsn]), "antenna": np.array([antenna]),
               "relays_there": np.array([relays_there]), "dv": np.array([dv])}
    env.expert = env.expert_plan()
    obs = env.observe()
    _, cont, disc, _, _ = agent.act(obs[0], deterministic=True, update_normalizer=False)
    a = env.decode(cont.numpy()[None], disc.numpy()[None])
    syn = env.synodic[target]
    tof = env.hohmann[target][2]
    return {"target": target, "objective": objective,
            "abort": bool(a["abort"][0]),
            "wait_s": float(a["wait"][0] * syn), "tof_s": float(a["tof"][0] * tof),
            "capture_alt_m": float(a["cap_alt"][0]), "aerocapture": bool(a["aero"][0]),
            "correction_at": float(a["corr_t"][0]),
            "relays": int(a["relays"][0]), "relay_orbit": RELAY_CLASSES[int(a["relay_class"][0])],
            "expert_dv": float(env.expert["total"][0]), "n": n,
            "reason": _reason(env, bool(a["abort"][0]), int(a["relays"][0]), dv)}


def _reason(env, abort: bool, relays: int, dv: float) -> str:
    """Человеческое объяснение решения навигатора."""
    need = float(env.expert["total"][0])
    if abort:
        if bool(env.expert["comm_impossible"][0]):
            return ("связь невозможна: станция слежения не достаёт до цели даже через "
                    "ретранслятор, а управление без связи запрещено — улучшите DSN")
        if dv < need:
            return f"Δv не хватит: нужно ≈{need:.0f} м/с, есть {dv:.0f}"
        return "отказ по оценке риска"
    if relays:
        return (f"связи с домом не хватит — сначала доставить {relays} "
                f"ретранслятор(а), затем основная миссия")
    return f"хватает: нужно ≈{need:.0f} м/с из {dv:.0f}"
