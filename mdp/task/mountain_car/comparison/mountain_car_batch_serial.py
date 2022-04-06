from __future__ import annotations

from mdp import common

from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.comparison.settings import Settings
# from mdp.model.non_tabular.feature.tile_coding.tiling_coding_parameters import TileCodingParameters
# from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters
# from mdp.task.mountain_car.enums import Dim


class MountainCarBatchSerial(ComparisonBuilder,
                             comparison_type=common.ComparisonType.MOUNTAIN_CAR_BATCH_SERIAL):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(
                training_episodes=20,
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_BATCH,
                    alpha=0.5
                ),
                value_function_parameters=common.ValueFunctionParameters(
                    value_function_type=common.ValueFunctionType.LINEAR_STATE_ACTION,
                    requires_delta_w=True
                )
            ),
            graph3d_values=self._graph3d_values
        )
