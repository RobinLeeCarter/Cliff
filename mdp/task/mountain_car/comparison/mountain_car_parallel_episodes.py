from __future__ import annotations

from mdp import common

from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.comparison.settings import Settings


class MountainCarParallelEpisodes(ComparisonBuilder,
                                  comparison_type=common.ComparisonType.MOUNTAIN_CAR_PARALLEL_EPISODES):
    def create(self) -> Comparison:
        # is the problem here that the other processes are in some sense off-policy?
        # in that the policy diverges from the last branching point
        #
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(
                training_episodes=10000,
                episodes_per_batch=800,
                episode_print_frequency=100,
                episode_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_PARALLEL_EPISODES,
                    alpha=0.5
                ),
                value_function_parameters=common.ValueFunctionParameters(
                    value_function_type=common.ValueFunctionType.LINEAR_STATE_ACTION,
                    requires_delta_w=True
                )
            ),
            graph3d_values=self._graph3d_values
        )
