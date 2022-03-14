from __future__ import annotations
from abc import ABC

from mdp.scenario.general_comparison_builder import GeneralComparisonBuilder


class ComparisonBuilder(GeneralComparisonBuilder, ABC):
    def __init__(self, random_wind: bool = False):
        super().__init__()
        self._random_wind: bool = random_wind
