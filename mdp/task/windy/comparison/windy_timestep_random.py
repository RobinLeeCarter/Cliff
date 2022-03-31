from __future__ import annotations

from mdp import common
from mdp.task.windy.comparison.comparison_builder import ComparisonBuilder
from mdp.task.windy.comparison.comparison import Comparison
from mdp.task.windy.model.environment_parameters import EnvironmentParameters


class WindyTimestepRandom(ComparisonBuilder,
                          comparison_type=common.ComparisonType.WINDY_TIMESTEP_RANDOM):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(random_wind=True),
            comparison_settings=self._settings,
            breakdown_parameters=self._breakdown_parameters,
            graph2d_values=self._graph2d_values,
            grid_view_parameters=self._grid_view_parameters
        )
