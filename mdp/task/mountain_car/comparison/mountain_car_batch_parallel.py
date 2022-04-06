from __future__ import annotations

from mdp import common

from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.comparison.settings import Settings


class MountainCarBatchParallel(ComparisonBuilder,
                               comparison_type=common.ComparisonType.MOUNTAIN_CAR_BATCH_PARALLEL):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(
                training_episodes=16,
                episodes_per_batch=8,
                episode_print_frequency=8,
                episode_multiprocessing=common.ParallelContextType.FORK_PICKLE,
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_BATCH
                ),
                value_function_parameters=common.ValueFunctionParameters(
                    value_function_type=common.ValueFunctionType.LINEAR_STATE_ACTION,
                    requires_delta_w=True
                )
            ),
            # graph3d_values=self._graph3d_values
        )
