from __future__ import annotations

from mdp import common

from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.comparison.settings import Settings


class MountainCarStandard(ComparisonBuilder,
                          comparison_type=common.ComparisonType.MOUNTAIN_CAR_STANDARD):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(
                training_episodes=9000
            ),
            graph3d_values=self._graph3d_values
        )
