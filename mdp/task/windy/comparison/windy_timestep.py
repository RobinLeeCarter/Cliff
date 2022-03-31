from __future__ import annotations

from mdp import common
from mdp.task.windy.comparison.comparison_builder import ComparisonBuilder
from mdp.task.windy.comparison.comparison import Comparison
from mdp.task.windy.model.environment_parameters import EnvironmentParameters


class WindyTimestep(ComparisonBuilder,
                    comparison_type=common.ComparisonType.WINDY_TIMESTEP):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=self._settings,
            breakdown_parameters=self._breakdown_parameters,
            graph2d_values=self._graph2d_values,
            grid_view_parameters=self._grid_view_parameters
        )
