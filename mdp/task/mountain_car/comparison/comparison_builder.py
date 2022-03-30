from __future__ import annotations
from abc import ABC

from mdp.task.base_comparison_builder import BaseComparisonBuilder
from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters


class ComparisonBuilder(BaseComparisonBuilder, ABC):
    def __init__(self):
        super().__init__()
        self._environment_parameters: EnvironmentParameters = EnvironmentParameters()
