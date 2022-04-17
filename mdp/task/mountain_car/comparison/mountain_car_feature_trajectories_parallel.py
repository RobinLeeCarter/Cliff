from __future__ import annotations

from mdp import common

from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.comparison.settings import Settings


class MountainCarFeatureTrajectoriesParallel(
    ComparisonBuilder,
    comparison_type=common.ComparisonType.MOUNTAIN_CAR_FEATURE_TRAJECTORIES_PARALLEL
        ):
    def create(self) -> Comparison:
        # is the problem here that the other processes are in some sense off-policy?
        # in that the policy diverges from the last branching point
        #
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(
                training_episodes=8000,
                episodes_per_batch=80,
                episode_print_frequency=100,
                episode_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.NT_SARSA_FEATURE_TRAJECTORIES_PARALLEL,
                    alpha=0.3
                ),
            ),
            graph3d_values=self._graph3d_values
        )
