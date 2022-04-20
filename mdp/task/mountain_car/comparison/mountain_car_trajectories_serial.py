from __future__ import annotations

from mdp import common

from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.comparison.settings import Settings
# from mdp.model.non_tabular.feature.tile_coding.tiling_coding_parameters import TileCodingParameters
# from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters
# from mdp.task.mountain_car.enums import Dim


class MountainCarTrajectoriesSerial(ComparisonBuilder,
                                    comparison_type=common.ComparisonType.MOUNTAIN_CAR_TRAJECTORIES_SERIAL):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(
                training_episodes=1000,
                episodes_per_batch=8,
                episode_print_frequency=100,
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.NT_SARSA_TRAJECTORIES_SERIAL,
                    alpha=0.5
                ),
            ),
            graph3d_values=self._graph3d_values
        )
