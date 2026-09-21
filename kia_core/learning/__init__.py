"""Профиль полёта и учёт вех. Обучения здесь нет: веса приходят готовыми."""
from .genome import FlightGenome, Individual
from .reward_book import REWARD_BOOK, FlightMetrics, RewardBook
from .rewards import Failure, Milestone, RewardSystem, ScoreEvent
