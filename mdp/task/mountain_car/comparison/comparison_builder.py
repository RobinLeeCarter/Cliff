from __future__ import annotations
from abc import ABC

from mdp import common
from mdp.task.base_comparison_builder import BaseComparisonBuilder
from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters


class ComparisonBuilder(BaseComparisonBuilder, ABC):
    def __init__(self):
        super().__init__()
        self._environment_parameters: EnvironmentParameters = EnvironmentParameters()
        self._graph3d_values = common.Graph3DValues(
                x_label="Position",
                y_label="Velocity",
                z_label="Time to go",
                steps=30
            )
