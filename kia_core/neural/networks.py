"""Нейросети KIA: actor-critic для PPO.

Архитектура (одинаковая для пилота и конструктора, отличаются головы):

    вход (сырая телеметрия) -> нормализация бегущим средним
      -> Linear(obs, hidden) -> Tanh
      -> Linear(hidden, hidden) -> Tanh
      -> [голова политики]  mu (непрерывные оси) + log_std (обучаемый)
         [дискретные головы] логиты (например, «отделить ступень»)
         [голова критика]   V(s) — оценка ожидаемой суммарной награды

Tanh, а не ReLU: для управления важна гладкость выходов — ReLU даёт
рваные скачки команд, которые в KSP выглядят как дёрганье рулями.
Ортогональная инициализация с малым усилением на голове политики —
стандартный приём PPO: первые действия близки к нулю и не разносят
аппарат на первом же шаге.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def orthogonal_init(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class RunningNormalizer(nn.Module):
    """Бегущие среднее и дисперсия входов (алгоритм Уэлфорда).

    Телеметрия KSP имеет разные масштабы и меняющиеся распределения
    (на старте скорость 0, на орбите 2300 м/с). Без нормализации
    градиенты по «мелким» признакам тонут в шуме «крупных».
    """

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def reset(self) -> None:
        """Забыть накопленную статистику входов.

        Нужно после того, как в неё попали выбросы. Один такой случай
        стоил проекту всего обучения: на гиперболической траектории
        симулятор отдавал апоапсис 1e9, признак получался 10 000 при
        рабочем диапазоне 0-1, и дисперсия по нему уходила в десятки
        тысяч. Дальше весь полезный диапазон апоапсиса, периапсиса и
        «прогресса к цели» умещался в 0.00005 сигмы — сеть переставала
        отличать 0 км от 100 км по этим входам. Статистика накопительная
        и живёт в чекпоинте, поэтому сама она уже не выправится: при 42
        млн наблюдений новый миллион сдвинет среднее на два процента.
        """
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.fill_(1e-4)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(normalized, -self.clip, self.clip)


class ActorCritic(nn.Module):
    """Общий ствол + головы политики (непрерывной и дискретных) и критика."""

    # Пределы разброса действий. Без верхней границы бонус за энтропию
    # раздувает log_std: политика становится всё случайнее, награда падает.
    # Именно это наблюдалось на прогоне 2 млн шагов (энтропия 7.19 -> 7.51).
    #
    # ВАЖНО про сам ограничитель. Раньше он стоял ТОЛЬКО в `distributions`
    # через torch.clamp — и это была ловушка: clamp обнуляет градиент за
    # пределами диапазона. Стоило параметру один раз вылезти за потолок,
    # как обратный путь закрывался навсегда: σ замерзала, политика больше
    # не могла стать точнее, сколько бы миллионов шагов ни прошло. Именно
    # так и случилось на прогоне 615 обновлений: log_std застрял на +0.201
    # (старый потолок), σ = 0.741, и сеть до конца рулила почти случайно.
    # Теперь параметр возвращается в диапазон ПРЯМО В СЕБЕ после каждого
    # шага оптимизатора (`clamp_log_std`), поэтому градиент жив всегда.
    LOG_STD_MIN = -2.5            # σ = 0.082, предел точности
    LOG_STD_MAX = -0.7            # σ = 0.497, предел разброса

    def __init__(self, obs_dim: int, continuous_dim: int = 0,
                 discrete_sizes: tuple[int, ...] = (), hidden: int = 256,
                 depth: int = 2, log_std_init: float = -0.2):
        super().__init__()
        self.obs_dim = obs_dim
        self.continuous_dim = continuous_dim
        self.discrete_sizes = tuple(discrete_sizes)
        self.normalizer = RunningNormalizer(obs_dim)

        trunk: list[nn.Module] = []
        last = obs_dim
        for _ in range(depth):
            trunk.append(orthogonal_init(nn.Linear(last, hidden)))
            trunk.append(nn.Tanh())
            last = hidden
        self.trunk = nn.Sequential(*trunk)

        if continuous_dim:
            self.mu_head = orthogonal_init(nn.Linear(hidden, continuous_dim),
                                           gain=0.01)
            # log_std не зависит от состояния — так PPO устойчивее на
            # управлении, чем при обучаемой гетероскедастичности
            self.log_std = nn.Parameter(torch.full((continuous_dim,), log_std_init))
        else:
            self.mu_head = None
            self.log_std = None

        self.discrete_heads = nn.ModuleList([
            orthogonal_init(nn.Linear(hidden, size), gain=0.01)
            for size in self.discrete_sizes
        ])
        self.value_head = orthogonal_init(nn.Linear(hidden, 1), gain=1.0)
        self.clamp_log_std()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def clamp_log_std(self) -> None:
        """Возвращает log_std в допустимый диапазон прямо в параметре.

        Вызывается после каждого шага оптимизатора и после загрузки весов.
        Смысл — держать параметр ВНУТРИ диапазона, а не обрезать его на
        выходе: обрезка на выходе убивает градиент и запирает разброс
        действий в потолке.
        """
        if self.log_std is None:
            return
        self.log_std.clamp_(self.LOG_STD_MIN, self.LOG_STD_MAX)

    # ------------------------------------------------------------------
    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.normalizer(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(obs)).squeeze(-1)

    # ------------------------------------------------------------------
    def distributions(self, obs: torch.Tensor):
        h = self.features(obs)
        normal = None
        if self.mu_head is not None:
            mu = torch.tanh(self.mu_head(h))          # держим mu в [-1, 1]
            log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = torch.exp(log_std).expand_as(mu)
            normal = Normal(mu, std)
        categoricals = []
        for head in self.discrete_heads:
            logits = head(h)
            if hasattr(self, "_action_mask") and self._action_mask is not None:
                logits = logits.masked_fill(~self._action_mask, float("-inf"))
            categoricals.append(Categorical(logits=logits))
        value = self.value_head(h).squeeze(-1)
        return normal, categoricals, value

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False,
            action_mask: torch.Tensor | None = None):
        """Выбор действия. Возвращает (continuous, discrete, logp, value)."""
        self._action_mask = action_mask
        normal, categoricals, value = self.distributions(obs)
        self._action_mask = None

        logp = torch.zeros(obs.shape[0] if obs.dim() > 1 else 1,
                           device=obs.device)
        continuous = torch.zeros((logp.shape[0], 0), device=obs.device)
        if normal is not None:
            continuous = normal.mean if deterministic else normal.sample()
            continuous = torch.clamp(continuous, -1.0, 1.0)
            logp = logp + normal.log_prob(continuous).sum(-1)

        discrete = []
        for dist in categoricals:
            choice = dist.probs.argmax(-1) if deterministic else dist.sample()
            logp = logp + dist.log_prob(choice)
            discrete.append(choice)
        discrete_tensor = (torch.stack(discrete, dim=-1) if discrete
                           else torch.zeros((logp.shape[0], 0),
                                            dtype=torch.long, device=obs.device))
        return continuous, discrete_tensor, logp, value

    # ------------------------------------------------------------------
    def evaluate(self, obs: torch.Tensor, continuous: torch.Tensor,
                 discrete: torch.Tensor, action_mask: torch.Tensor | None = None):
        """Лог-вероятности и энтропия для уже сделанных действий (шаг PPO)."""
        self._action_mask = action_mask
        normal, categoricals, value = self.distributions(obs)
        self._action_mask = None

        logp = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)
        if normal is not None and continuous.shape[-1] > 0:
            logp = logp + normal.log_prob(continuous).sum(-1)
            entropy = entropy + normal.entropy().sum(-1)
        for index, dist in enumerate(categoricals):
            choice = discrete[:, index]
            logp = logp + dist.log_prob(choice)
            entropy = entropy + dist.entropy()
        return logp, entropy, value

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        heads = []
        if self.continuous_dim:
            heads.append(f"{self.continuous_dim} непрерывных осей")
        if self.discrete_sizes:
            heads.append("дискретные головы " + "×".join(map(str, self.discrete_sizes)))
        return (f"ActorCritic({self.obs_dim} входов -> "
                f"{', '.join(heads) or 'только критик'}), "
                f"{self.parameter_count():,} весов".replace(",", " "))
