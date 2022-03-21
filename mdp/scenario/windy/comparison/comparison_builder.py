from __future__ import annotations
from abc import ABC

from mdp.scenario.base_comparison_builder import BaseComparisonBuilder


class ComparisonBuilder(BaseComparisonBuilder, ABC):
    def __init__(self, random_wind: bool = False):
        super().__init__()
        self._random_wind: bool = random_wind
