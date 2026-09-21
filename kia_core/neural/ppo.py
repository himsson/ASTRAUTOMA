"""PPO (Proximal Policy Optimization) — ядро обучения KIA.

Реализация классического PPO-clip (Schulman et al., 2017) со всеми
приёмами, без которых он не работает на управлении:

* GAE(λ) для оценки преимущества;
* обрезка отношения вероятностей (clip) и обрезка функции ценности;
* нормализация преимуществ внутри мини-батча;
* бонус за энтропию (иначе политика схлопывается в одно действие);
* обрезка нормы градиента;
* ранняя остановка эпохи по KL-дивергенции;
* линейный отжиг learning rate.

Веса, состояние оптимизатора, статистика нормализации входов и счётчики
обучения сохраняются в kia_model.pth и загружаются при следующем запуске.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn

from ..config import DATA_DIR, ROOT
from ..logging_setup import get_logger
from .device import get_device, gpu_memory_report
from .networks import ActorCritic

log = get_logger("neural.ppo")

MODEL_FILE = ROOT / "kia_model.pth"
BEST_MODEL_FILE = ROOT / "kia_model_best.pth"
# Снимки ступеней учебной программы: см. neural/curriculum.py
STAGE_FILES = {1: ROOT / "kia_model_stage1.pth",
               2: ROOT / "kia_model_stage2.pth",
               3: ROOT / "kia_model_stage3.pth"}


# Скорость обучения по ступеням учебной программы.
#
# На первой ступени сеть учится с нуля — там нужен размашистый шаг.
# Дальше она НЕ учится заново: она доучивает уже работающие навыки под
# новые правила мира. Крупный шаг градиента в этой ситуации ломает
# базовое умение выводить аппарат на орбиту быстрее, чем успевает
# научить чему-то новому, — ровно это и наблюдалось обвалом политики на
# 112-м обновлении Тренажёра+.
# Числа для первой ступени не подобраны, а ЗАМЕРЕНЫ. Опыт: та же среда,
# обучение с чистого листа, 1.5 млн шагов при LR 1.5e-4, σ 0.165 и бонусе
# за разведку 0.0003 — 164 орбиты из 1098 эпизодов за четыре минуты.
# Прежние «широкие» настройки первой ступени (LR 3.0e-4, σ 0.368, бонус
# 0.002) достались ей от старого плоского мира и в настоящем мире дают
# другое: сеть учится НЕ ВЗЛЕТАТЬ. Живой замер — средняя команда газа
# ползла −0.05 -> −0.37 за пять минут, орбиты в обучении делал разброс, а
# детерминированный прогон давал ровно −2.1, то есть стоянку на столе.
STAGE_LEARNING_RATE = {
    1: 1.5e-4,      # обучение с нуля в настоящем мире
    2: 1.5e-4,      # доучивание под скрытые механики
    3: 1.5e-4,      # втрое более частые решения
    4: 1.0e-4,      # смена закона управления — самый деликатный переход
}

# Автономный конвейер (neural/pipeline.py) снижает скорость обучения той
# ступени, на которой политика зашла в тупик, — вдвое на каждую неудачную
# попытку. Поправки лежат отдельным файлом, а не переписывают этот модуль:
# самомодифицирующийся код при обрыве записи оставил бы систему без ppo.py.
LR_OVERRIDE_FILE = DATA_DIR / "stage_learning_rate.json"

# Ниже этого значения снижать бессмысленно: сеть перестаёт учиться вообще
MIN_STAGE_LEARNING_RATE = 1.0e-5


def _lr_overrides() -> dict:
    try:
        raw = json.loads(LR_OVERRIDE_FILE.read_text(encoding="utf-8"))
        return {str(key): float(value) for key, value in raw.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


# Бонус за энтропию по ступеням.
#
# На первой ступени разведка нужна: сеть учится с нуля и должна пробовать
# всё подряд. Дальше — ровно наоборот. Управление моментом на Тренажёре++
# требует ТОЧНЫХ команд, а бонус за энтропию платит за их разброс. На
# прогоне 615 обновлений это кончилось тем, что политика вообще не
# сошлась: детерминированно она шла строго вверх, а все орбиты делал шум.
STAGE_ENTROPY_COEF = {
    1: 0.0003,      # столько же, сколько в удавшемся замере с нуля
    2: 0.0003,      # скрытые механики
    3: 0.0003,      # втрое более частые решения
    4: 0.0002,      # управление моментом — политика обязана сойтись
}

# Потолок разброса действий по ступеням.
#
# Замер, ради которого это появилось: со снятой ловушкой градиента, но при
# потолке −0.7 (σ 0.50) сеть за 480 тыс. шагов НЕ опустила разброс ни на
# сколько — бонус за энтропию всё равно перевешивал. А пока σ велика,
# обучается не политика, а распределение вокруг неё: детерминированный
# прогон показывает совсем другое поведение, чем обучающий. Именно так
# две ступени и были «сданы» шумом.
#
# Лечится потолком. При σ 0.13 выбранное действие практически совпадает
# со средним, и обучать распределение = обучать политику. На первой
# ступени разброс ещё нужен для разведки, дальше он только мешает.
#
# ВАЖНО, КАК ЭТО РАБОТАЕТ НА ПРАКТИКЕ. Бонус за энтропию, даже мизерный,
# всегда прижимает log_std к потолку: сигнал политики к этому параметру
# слабый (он идёт через корреляцию преимущества с квадратом отклонения),
# а бонус давит постоянно. Замерено дважды — при потолке −0.7 и при −1.0
# энтропия вставала ровно на потолочное значение. Поэтому потолок здесь
# не «предохранитель на крайний случай», а РАБОЧЕЕ значение σ, которым
# управляют напрямую. Так честнее и предсказуемее, чем надеяться, что
# сеть подберёт разброс сама.
# Первая ступень получила −1.8 не из соображений красоты: именно при этом
# потолке сеть выучила настоящий мир с чистого листа за четыре минуты.
# Прежние −1.0 (σ 0.368) годились для плоской игрушечной ступени, а здесь
# давали классику раздела 5.3d: обучается распределение ВОКРУГ политики,
# а не сама политика — орбиты делал шум, средняя команда не взлетала.
STAGE_LOG_STD_MAX = {
    1: -1.80,       # σ 0.165 — замерено, учит с нуля
    2: -1.85,       # σ 0.157
    3: -1.90,       # σ 0.150 — решения втрое чаще, нужна точность
    4: -2.00,       # σ 0.135 — точное управление моментом
}


def log_std_max_for_stage(stage: int) -> float:
    return STAGE_LOG_STD_MAX.get(int(stage), -1.0)


def learning_rate_for_stage(stage: int) -> float:
    """Скорость обучения ступени с учётом накопленных понижений."""
    base = STAGE_LEARNING_RATE.get(int(stage), 3.0e-4)
    return _lr_overrides().get(str(int(stage)), base)


def entropy_coef_for_stage(stage: int) -> float:
    return STAGE_ENTROPY_COEF.get(int(stage), 0.001)


def scale_stage_learning_rate(stage: int, factor: float) -> float:
    """Умножает скорость обучения ступени и запоминает это на диске."""
    current = learning_rate_for_stage(stage)
    updated = max(MIN_STAGE_LEARNING_RATE, current * float(factor))
    overrides = _lr_overrides()
    overrides[str(int(stage))] = updated
    LR_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LR_OVERRIDE_FILE.write_text(json.dumps(overrides, indent=2),
                                encoding="utf-8")
    log.info("Скорость обучения ступени %d: %.2e -> %.2e", stage, current,
             updated)
    return updated


def reset_stage_learning_rate(stage: int | None = None) -> None:
    """Возвращает исходные скорости обучения (все или одной ступени)."""
    if stage is None:
        LR_OVERRIDE_FILE.unlink(missing_ok=True)
        return
    overrides = _lr_overrides()
    if overrides.pop(str(int(stage)), None) is not None:
        LR_OVERRIDE_FILE.write_text(json.dumps(overrides, indent=2),
                                    encoding="utf-8")


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.999          # шаг 0.2 с: горизонт ≈ 1000 шагов ≈ 200 с полёта
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip: float = 0.2
    # 0.01 перевешивал сигнал политики, 0.003 всё ещё выталкивал разброс
    # действий в потолок ограничителя (энтропия 7.06 при максимуме 7.1)
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch_size: int = 256
    target_kl: float = 0.03
    anneal_lr: bool = True
    total_updates_planned: int = 1000
    # С какого номера обновления считать отжиг. Счётчик `stats.updates`
    # накопительный за всю жизнь мозга, а план — на один прогон. Без этой
    # отметки продолженное обучение стартовало сразу с конца расписания:
    # 185 прошлых обновлений против 48 запланированных давали прогресс
    # 100 %, и скорость обучения мгновенно падала до одной десятой.
    anneal_baseline: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingStats:
    updates: int = 0
    total_steps: int = 0
    episodes: int = 0
    best_return: float = float("-inf")
    last_returns: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"updates": self.updates, "total_steps": self.total_steps,
                "episodes": self.episodes,
                "best_return": (self.best_return
                                if self.best_return > float("-inf") else None),
                "last_returns": self.last_returns[-50:],
                "history": self.history[-500:]}

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingStats":
        stats = cls()
        stats.updates = int(data.get("updates", 0))
        stats.total_steps = int(data.get("total_steps", 0))
        stats.episodes = int(data.get("episodes", 0))
        best = data.get("best_return")
        stats.best_return = float(best) if best is not None else float("-inf")
        stats.last_returns = list(data.get("last_returns", []))
        stats.history = list(data.get("history", []))
        return stats


# ==========================================================================
class RolloutBuffer:
    """Хранилище одного пакета взаимодействий со средой."""

    def __init__(self, obs_dim: int, continuous_dim: int, discrete_count: int,
                 capacity: int, device, mask_dim: int = 0):
        self.device = device
        self.capacity = capacity
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.continuous = torch.zeros((capacity, continuous_dim), device=device)
        self.discrete = torch.zeros((capacity, discrete_count),
                                    dtype=torch.long, device=device)
        self.logp = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        # Маски запрещённых действий хранятся вместе с шагом. Без этого
        # обучение видит не то распределение, из которого действие взято:
        # при выборе часть вариантов была запрещена, а при обновлении —
        # снова разрешена, и отношение вероятностей PPO становится враньём.
        self.mask_dim = mask_dim
        self.masks = (torch.ones((capacity, mask_dim), dtype=torch.bool,
                                 device=device) if mask_dim else None)
        self.has_masks = False
        self.size = 0

    def add(self, obs, continuous, discrete, logp, value, reward, done,
            mask=None) -> None:
        if self.size >= self.capacity:
            return
        i = self.size
        self.obs[i] = obs
        if continuous is not None and continuous.numel():
            self.continuous[i] = continuous
        if discrete is not None and discrete.numel():
            self.discrete[i] = discrete
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        if self.masks is not None:
            if mask is None:
                self.masks[i] = True
            else:
                self.masks[i] = torch.as_tensor(mask, dtype=torch.bool,
                                                device=self.device)
                self.has_masks = True
        self.size += 1

    @property
    def full(self) -> bool:
        return self.size >= self.capacity

    def clear(self) -> None:
        self.size = 0
        self.has_masks = False

    # ------------------------------------------------------------------
    def compute_returns(self, last_value: float, gamma: float, lam: float):
        """GAE(λ): сглаженная оценка преимущества каждого шага."""
        n = self.size
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0
        next_value = float(last_value)
        for t in reversed(range(n)):
            non_terminal = 1.0 - self.dones[t]
            delta = (self.rewards[t] + gamma * next_value * non_terminal
                     - self.values[t])
            last_gae = delta + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]
        returns = advantages + self.values[:n]
        return advantages, returns


# ==========================================================================
class VectorRolloutBuffer(RolloutBuffer):
    """Буфер для пачки сред: всё хранится как (шаг, копия).

    Отличие от обычного буфера одно, но важное: преимущества считаются
    ОТДЕЛЬНО по каждой копии. Копии живут своей жизнью и заканчивают
    эпизоды в разное время; если склеить их в одну ленту, награда одной
    ракеты потечёт в оценку другой, и обучение будет считать по чужим
    данным.
    """

    def __init__(self, obs_dim: int, continuous_dim: int, discrete_count: int,
                 capacity: int, device, envs: int, mask_dim: int = 0):
        self.envs = max(1, int(envs))
        self.device = device
        self.capacity = capacity
        shape = (capacity, self.envs)
        self.obs = torch.zeros(shape + (obs_dim,), device=device)
        self.continuous = torch.zeros(shape + (continuous_dim,), device=device)
        self.discrete = torch.zeros(shape + (discrete_count,),
                                    dtype=torch.long, device=device)
        self.logp = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.dones = torch.zeros(shape, device=device)
        self.mask_dim = mask_dim
        self.masks = (torch.ones(shape + (mask_dim,), dtype=torch.bool,
                                 device=device) if mask_dim else None)
        self.has_masks = False
        self.size = 0

    # ------------------------------------------------------------------
    def add_batch(self, obs, continuous, discrete, logp, value, rewards,
                  dones, masks=None) -> None:
        if self.size >= self.capacity:
            return
        i = self.size
        self.obs[i] = obs
        if continuous is not None and continuous.numel():
            self.continuous[i] = continuous
        if discrete is not None and discrete.numel():
            self.discrete[i] = discrete
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = torch.as_tensor(rewards, dtype=torch.float32,
                                          device=self.device)
        self.dones[i] = torch.as_tensor(dones, dtype=torch.float32,
                                        device=self.device)
        if self.masks is not None:
            if masks is None:
                self.masks[i] = True
            else:
                self.masks[i] = torch.as_tensor(masks, dtype=torch.bool,
                                                device=self.device)
                self.has_masks = True
        self.size += 1

    @property
    def transitions(self) -> int:
        """Сколько всего переходов накоплено (шаги × копии)."""
        return self.size * self.envs

    # ------------------------------------------------------------------
    def compute_returns(self, last_value, gamma: float, lam: float):
        """GAE(λ) по каждой копии отдельно, затем всё вытягивается в ленту."""
        n = self.size
        advantages = torch.zeros((n, self.envs), device=self.device)
        last_gae = torch.zeros(self.envs, device=self.device)
        next_value = torch.as_tensor(last_value, dtype=torch.float32,
                                     device=self.device).reshape(-1)
        if next_value.numel() == 1:
            next_value = next_value.expand(self.envs).clone()
        for t in reversed(range(n)):
            non_terminal = 1.0 - self.dones[t]
            delta = (self.rewards[t] + gamma * next_value * non_terminal
                     - self.values[t])
            last_gae = delta + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]
        returns = advantages + self.values[:n]
        return advantages.reshape(-1), returns.reshape(-1)

    # ------------------------------------------------------------------
    def flat(self):
        """Данные пачки в виде плоской ленты для мини-батчей."""
        n = self.size
        return {
            "obs": self.obs[:n].reshape(n * self.envs, -1),
            "continuous": self.continuous[:n].reshape(n * self.envs, -1),
            "discrete": self.discrete[:n].reshape(n * self.envs, -1),
            "logp": self.logp[:n].reshape(-1),
            "values": self.values[:n].reshape(-1),
            "masks": (self.masks[:n].reshape(n * self.envs, -1)
                      if self.has_masks else None),
        }


# ==========================================================================
class PPOAgent:
    """Обучаемый агент: сеть + оптимизатор + правило обновления."""

    def __init__(self, obs_dim: int, continuous_dim: int = 0,
                 discrete_sizes: tuple[int, ...] = (), config: PPOConfig | None = None,
                 hidden: int = 256, depth: int = 2, name: str = "agent"):
        self.name = name
        self.config = config or PPOConfig()
        self.device = get_device()
        self.net = ActorCritic(obs_dim, continuous_dim, discrete_sizes,
                               hidden=hidden, depth=depth).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.config.learning_rate, eps=1e-5)
        self.stats = TrainingStats()
        self.obs_dim = obs_dim
        self.continuous_dim = continuous_dim
        self.discrete_sizes = tuple(discrete_sizes)
        # Гибридный вывод подключается только для сети пилота: у неё
        # пачки достаточно большие, чтобы делить их имело смысл
        self.hybrid = None
        log.info("[%s] %s на %s", name, self.net.describe(), self.device)

    def enable_hybrid(self, share: float | None = None) -> bool:
        """Включает совместный счёт процессора и видеокарты."""
        from .hybrid import HybridActor
        self.hybrid = HybridActor(self.net, share=share)
        return self.hybrid.enabled

    # ------------------------------------------------------------------
    def observe(self, obs_vector: list[float]) -> torch.Tensor:
        return torch.as_tensor(obs_vector, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

    @torch.no_grad()
    def act(self, obs_vector: list[float], deterministic: bool = False,
            action_mask=None, update_normalizer: bool = True):
        obs = self.observe(obs_vector)
        if update_normalizer:
            # Проверочные прогоны нормализатор не обновляют: иначе экзамен
            # менял бы статистику входов, по которой считает обучение
            self.net.normalizer.update(obs)
        mask = None
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool,
                                   device=self.device).unsqueeze(0)
        continuous, discrete, logp, value = self.net.act(obs, deterministic, mask)
        return (obs.squeeze(0), continuous.squeeze(0), discrete.squeeze(0),
                logp.squeeze(0), value.squeeze(0))

    @torch.no_grad()
    def act_many(self, observations, action_masks=None, while_waiting=None):
        """Решение сразу по пачке наблюдений — один проход сети на всех.

        Ради этого метода и сделана векторизация сред: расход на вызов
        сети почти не зависит от размера пачки, поэтому шестнадцать
        наблюдений за раз обходятся почти как одно.

        while_waiting передаётся дальше в совместный режим: это работа,
        которую процессор делает, пока видеокарта считает свою долю.
        Возвращает (наблюдения, действия…, результат while_waiting).
        """
        obs = torch.as_tensor(observations, dtype=torch.float32,
                              device=self.device)
        self.net.normalizer.update(obs)
        mask = None
        if action_masks is not None:
            mask = torch.as_tensor(action_masks, dtype=torch.bool,
                                   device=self.device)
        if self.hybrid is not None and self.hybrid.enabled:
            # часть пачки уходит на видеокарту и считается одновременно
            action, waited = self.hybrid.act(obs, mask, while_waiting)
        else:
            action = self.net.act(obs, False, mask)
            waited = while_waiting(action) if while_waiting else None
        continuous, discrete, logp, value = action
        return obs, continuous, discrete, logp, value, waited

    def hybrid_cpu_count(self, total: int) -> int:
        """Сколько копий обслуживает процессор (остальные — видеокарта)."""
        if self.hybrid is None or not self.hybrid.enabled:
            return total
        return self.hybrid.cpu_count(total)

    # ------------------------------------------------------------------
    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        """Один шаг обучения PPO по собранному пакету. Возвращает метрики."""
        if buffer.size < 2:
            return {"skipped": True, "reason": "мало данных"}

        cfg = self.config
        advantages, returns = buffer.compute_returns(last_value, cfg.gamma,
                                                     cfg.gae_lambda)
        if hasattr(buffer, "flat"):
            # пачка сред: (шаг, копия, …) вытянуто в ленту переходов
            data = buffer.flat()
            n = buffer.transitions
            obs = data["obs"]
            continuous = data["continuous"]
            discrete = data["discrete"]
            old_logp = data["logp"]
            old_values = data["values"]
            masks = data["masks"]
        else:
            n = buffer.size
            obs = buffer.obs[:n]
            continuous = buffer.continuous[:n]
            discrete = buffer.discrete[:n]
            old_logp = buffer.logp[:n]
            old_values = buffer.values[:n]
            masks = buffer.masks[:n] if buffer.has_masks else None

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if cfg.anneal_lr and cfg.total_updates_planned > 0:
            done = max(0, self.stats.updates - cfg.anneal_baseline)
            progress = min(1.0, done / cfg.total_updates_planned)
            lr = cfg.learning_rate * (1.0 - 0.9 * progress)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = cfg.learning_rate

        policy_losses, value_losses, entropies, kls, clipfracs = [], [], [], [], []
        indices = torch.arange(n, device=self.device)
        stop = False

        for epoch in range(cfg.epochs):
            permutation = indices[torch.randperm(n, device=self.device)]
            for start in range(0, n, cfg.minibatch_size):
                batch = permutation[start:start + cfg.minibatch_size]
                if batch.numel() < 2:
                    continue
                logp, entropy, value = self.net.evaluate(
                    obs[batch], continuous[batch], discrete[batch],
                    action_mask=None if masks is None else masks[batch])

                ratio = torch.exp(logp - old_logp[batch])
                surrogate_1 = ratio * advantages[batch]
                surrogate_2 = torch.clamp(ratio, 1 - cfg.clip_range,
                                          1 + cfg.clip_range) * advantages[batch]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_clipped = old_values[batch] + torch.clamp(
                    value - old_values[batch], -cfg.value_clip, cfg.value_clip)
                value_loss = torch.max((value - returns[batch]).pow(2),
                                       (value_clipped - returns[batch]).pow(2)).mean()

                entropy_bonus = entropy.mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                # держим разброс действий внутри диапазона В САМОМ
                # параметре: иначе градиент к нему обнуляется и политика
                # уже никогда не станет точнее (см. networks.LOG_STD_MAX)
                self.net.clamp_log_std()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clipfrac = ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())
                kls.append(approx_kl)
                clipfracs.append(clipfrac)

                if cfg.target_kl and approx_kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        self.stats.updates += 1
        self.stats.total_steps += n
        if self.hybrid is not None:
            # зеркало на видеокарте обязано повторять обученные веса,
            # иначе оно начнёт выдавать действия устаревшей политики
            self.hybrid.sync()
        metrics = {
            "steps": n,
            "policy_loss": _mean(policy_losses),
            "value_loss": _mean(value_losses),
            "entropy": _mean(entropies),
            "approx_kl": _mean(kls),
            "clip_fraction": _mean(clipfracs),
            "learning_rate": lr,
            "updates": self.stats.updates,
            "total_steps": self.stats.total_steps,
            "early_stop": stop,
        }
        self.stats.history.append({"update": self.stats.updates, **{
            k: round(v, 5) for k, v in metrics.items()
            if isinstance(v, (int, float))}})
        return metrics

    # ------------------------------------------------------------------
    def record_episode(self, episode_return: float) -> None:
        self.stats.episodes += 1
        self.stats.last_returns.append(round(float(episode_return), 2))
        self.stats.last_returns = self.stats.last_returns[-200:]
        if episode_return > self.stats.best_return:
            self.stats.best_return = float(episode_return)

    def mean_return(self, window: int = 20) -> float:
        recent = self.stats.last_returns[-window:]
        return sum(recent) / len(recent) if recent else 0.0

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats.to_dict(),
            "config": self.config.to_dict(),
            "obs_dim": self.obs_dim,
            "continuous_dim": self.continuous_dim,
            "discrete_sizes": list(self.discrete_sizes),
        }

    def load_state_dict(self, data: dict, strict: bool = True) -> bool:
        if data.get("obs_dim") != self.obs_dim:
            log.warning("[%s] размерность наблюдений изменилась (%s -> %s) — "
                        "веса не подходят, начинаем с нуля",
                        self.name, data.get("obs_dim"), self.obs_dim)
            return False
        try:
            self.net.load_state_dict(data["net"], strict=strict)
            self.optimizer.load_state_dict(data["optimizer"])
            self.stats = TrainingStats.from_dict(data.get("stats", {}))
            # Старые файлы могли сохранить log_std за потолком — тогда он
            # был заперт там навсегда. Чиним при загрузке, иначе плохой
            # чекпоинт продолжал бы тащить замороженный разброс действий.
            before = (float(self.net.log_std.max().item())
                      if self.net.log_std is not None else 0.0)
            self.net.clamp_log_std()
            after = (float(self.net.log_std.max().item())
                     if self.net.log_std is not None else 0.0)
            if before > after + 1e-6:
                log.warning("[%s] разброс действий был заперт за потолком "
                            "(log_std %.3f) — возвращён к %.3f (σ %.3f)",
                            self.name, before, after, math.exp(after))
            return True
        except Exception as exc:
            log.error("[%s] веса не загрузились: %s", self.name, exc)
            return False


# ==========================================================================
class Brain:
    """Мозг KIA: пилот + конструктор в одном файле весов kia_model.pth."""

    VERSION = 2

    def __init__(self, pilot: PPOAgent, builder: PPOAgent | None = None,
                 path: Path | None = None, mathematician: PPOAgent | None = None):
        self.pilot = pilot
        self.builder = builder
        # Третья сеть: счёт манёвров. Живёт в том же файле весов, но учится
        # отдельно — у неё своё пространство задач и свои ступени.
        self.mathematician = mathematician
        self.path = Path(path or MODEL_FILE)
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        self.loaded_from_disk = False
        self.best_path = Path(BEST_MODEL_FILE)
        self.best_mean = float("-inf")
        # Ступень учебной программы, на которой эти веса обучались.
        # Старые файлы поля не содержат — значит, базовый тренажёр.
        self.stage = 1
        # Ступени остальных программ. Пилотская лежит в self.stage —
        # так файлы, записанные до появления конструкторской и
        # математической программ, читаются без единой оговорки.
        self.stages: dict[str, int] = {"builder": 1, "math": 1}

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        target = Path(path or self.path)
        payload = {
            "version": self.VERSION,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created": self.created,
            "device": str(self.pilot.device),
            "stage": self.stage,
            "pilot": self.pilot.state_dict(),
        }
        if self.builder is not None:
            payload["builder"] = self.builder.state_dict()
        if self.mathematician is not None:
            payload["mathematician"] = self.mathematician.state_dict()
        # Ступени трёх сетей независимы: пилот может быть на четвёртой,
        # конструктор на второй, математик ещё на первой. Одним числом
        # `stage` их не описать — оно осталось за пилотом, а остальные
        # хранятся отдельно.
        payload["stages"] = dict(self.stages)
        tmp = target.with_suffix(".pth.tmp")
        torch.save(payload, tmp)
        tmp.replace(target)
        log.info("Веса сохранены: %s (обновлений: пилот %d, конструктор %s)",
                 target, self.pilot.stats.updates,
                 self.builder.stats.updates if self.builder else "—")
        return target

    def save_best(self, mean_return: float) -> bool:
        """Отдельная копия весов на момент лучшего среднего результата.

        Обучение с подкреплением умеет деградировать: политика,
        показавшая +60, через сотню обновлений может скатиться в шум и
        затереть себя в основном файле. Лучшая версия хранится отдельно
        и не перезаписывается ухудшениями.
        """
        if mean_return <= self.best_mean:
            return False
        self.best_mean = mean_return
        self.save(self.best_path)
        log.info("Новый лучший результат (средняя %.2f) — веса сохранены в %s",
                 mean_return, self.best_path.name)
        return True

    def normalizer_health(self) -> list[tuple[str, float]]:
        """Какие входы задавлены выбросами: [(признак, доля сигмы)].

        Для каждого признака считается, сколько сигм занимает его рабочий
        диапазон. Здоровый вход — единицы и доли единицы. Тысячные доли
        означают, что признак для сети превратился в константу.
        """
        from .spaces import FLIGHT_FEATURES

        normalizer = self.pilot.net.normalizer
        health = []
        for index, (name, _) in enumerate(FLIGHT_FEATURES):
            sigma = float(normalizer.var[index]) ** 0.5
            health.append((name, 1.0 / sigma if sigma > 0 else float("inf")))
        return health

    def reset_normalizer(self) -> None:
        """Обнуляет статистику входов пилота и конструктора."""
        self.pilot.net.normalizer.reset()
        if self.builder is not None:
            self.builder.net.normalizer.reset()
        log.warning("Статистика входов обнулена: сеть заново набирает "
                    "масштабы признаков")

    def restore_best(self) -> bool:
        """Возвращает лучшие веса в рабочий файл."""
        if not self.best_path.exists():
            log.warning("Файла лучших весов нет: %s", self.best_path)
            return False
        if self.load(self.best_path):
            self.save(self.path)
            log.info("Лучшие веса восстановлены в %s", self.path.name)
            return True
        return False

    def load(self, path: Path | None = None) -> bool:
        source = Path(path or self.path)
        if not source.exists():
            log.info("Файл весов %s не найден — мозг стартует с нуля", source.name)
            return False
        try:
            payload = torch.load(source, map_location=self.pilot.device,  # только тензоры: безопасно
                                 weights_only=True)
        except Exception as exc:
            log.error("Не удалось прочитать %s: %s", source.name, exc)
            return False
        ok = self.pilot.load_state_dict(payload.get("pilot", {}))
        if self.builder is not None and "builder" in payload:
            self.builder.load_state_dict(payload["builder"])
        if self.mathematician is not None and "mathematician" in payload:
            self.mathematician.load_state_dict(payload["mathematician"])
        self.created = payload.get("created", self.created)
        self.stage = int(payload.get("stage", 1))
        stored = payload.get("stages") or {}
        for key in ("builder", "math"):
            self.stages[key] = int(stored.get(key, 1))
        self.loaded_from_disk = ok
        if ok:
            log.info("Мозг загружен из %s: ступень %d, %d обновлений, "
                     "%d шагов, %d эпизодов, рекорд %.1f", source.name,
                     self.stage, self.pilot.stats.updates,
                     self.pilot.stats.total_steps, self.pilot.stats.episodes,
                     self.pilot.stats.best_return)
        return ok

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"Файл весов: {self.path}"
            + ("" if self.path.exists() else "  (ещё не создан)"),
            f"Ступень программы: {self.stage}",
            f"Устройство: {self.pilot.device}",
            f"Пилот:       {self.pilot.net.describe()}",
        ]
        if self.builder is not None:
            lines.append(f"Конструктор: {self.builder.net.describe()} "
                         f"(ступень {self.stages.get('builder', 1)})")
        if self.mathematician is not None:
            lines.append(f"Математик:   {self.mathematician.net.describe()} "
                         f"(ступень {self.stages.get('math', 1)})")
        stats = self.pilot.stats
        lines += [
            f"Обновлений весов: {stats.updates}",
            f"Шагов среды:      {stats.total_steps}",
            f"Эпизодов:         {stats.episodes}",
            f"Лучшая награда:   "
            f"{stats.best_return if stats.best_return > float('-inf') else '—'}",
            f"Средняя за 20:    {self.pilot.mean_return(20):.1f}",
            gpu_memory_report(),
        ]
        return "\n".join(lines)

    def metrics_json(self) -> str:
        return json.dumps({
            "pilot": self.pilot.stats.to_dict(),
            "builder": self.builder.stats.to_dict() if self.builder else None,
            "mathematician": (self.mathematician.stats.to_dict()
                              if self.mathematician else None),
            "stages": dict(self.stages),
        }, ensure_ascii=False, indent=2)


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


# ==========================================================================
def build_brain(config: PPOConfig | None = None, load: bool = True) -> Brain:
    """Собирает мозг KIA под текущие пространства и подгружает опыт."""
    from .spaces import (BUILDER_OBS_DIM, CONTINUOUS_DIM, DISCRETE_ACTIONS,
                         FLIGHT_OBS_DIM)

    pilot = PPOAgent(
        obs_dim=FLIGHT_OBS_DIM,
        continuous_dim=CONTINUOUS_DIM,
        discrete_sizes=tuple(size for _, size in DISCRETE_ACTIONS),
        config=config or PPOConfig(),
        hidden=256, depth=2, name="пилот")

    from .builder_env import BUILDER_ACTION_COUNT
    builder = PPOAgent(
        obs_dim=BUILDER_OBS_DIM,
        continuous_dim=0,
        discrete_sizes=(BUILDER_ACTION_COUNT,),
        config=config or PPOConfig(),
        hidden=128, depth=2, name="конструктор")

    from .math_env import MATH_ANSWER_DIM, MATH_OBS_DIM
    mathematician = PPOAgent(
        obs_dim=MATH_OBS_DIM,
        continuous_dim=MATH_ANSWER_DIM,
        discrete_sizes=(),
        config=config or PPOConfig(),
        # Три слоя, а не два. Логарифмы входов делают линейными формулы
        # вида sqrt(mu/r), но не все: в Циолковском ответ зависит от
        # ЛОГАРИФМА отношения масс, то есть сети приходится применить
        # логарифм к собственной внутренней величине. Одним скрытым слоем
        # это не приближается — замер показал 35 % против 92 % на соседних
        # задачах того же уровня.
        hidden=256, depth=3, name="математик")

    brain = Brain(pilot, builder, mathematician=mathematician)
    if load:
        brain.load()
    # Пилот считается пачками по 16 и больше — есть что делить с видеокартой
    pilot.enable_hybrid()
    return brain
