from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.task.random_walk.comparison.comparison_builder import ComparisonBuilder
from mdp.task.random_walk.comparison.comparison import Comparison
from mdp.task.random_walk.model.environment_parameters import EnvironmentParameters


@dataclass(unsafe_hash=True)    # needed for multiprocessing where results may differ, potentially pickle
class AlgorithmParameters(common.AlgorithmParameters):
    initial_v_value: float = 0.5


@dataclass
class Settings(common.Settings):
    runs: int = 100
    # runs_multiprocessing: common.ParallelContextType =common.ParallelContextType.FORK
    training_episodes: int = 100
    policy_parameters: common.PolicyParameters = common.PolicyParameters(
        policy_type=common.PolicyType.TABULAR_NONE
    )
    algorithm_parameters: common.AlgorithmParameters = AlgorithmParameters()


class RandomWalkEpisode(ComparisonBuilder,
                        comparison_type=common.ComparisonType.RANDOM_WALK_EPISODE):
    def create(self):
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=Settings(),
            breakdown_parameters=common.BreakdownParameters(
                breakdown_type=common.BreakdownType.RMS_BY_EPISODE,
            ),
            settings_list=[
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_TD_0,
                        alpha=0.05,
                    )
                ),
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_TD_0,
                        alpha=0.1,
                    )
                ),
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_TD_0,
                        alpha=0.15,
                    )
                ),
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.01,
                    )
                ),
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.02,
                    )
                ),
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.03,
                    )
                ),
                Settings(
                    algorithm_parameters=AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.04,
                    )
                ),
            ],
            settings_list_multiprocessing=common.ParallelContextType.FORK,
            graph2d_values=common.Graph2DValues(
                has_grid=True,
                has_legend=True,
                y_min=0.0,
                y_max=0.25
            ),
            grid_view_parameters=common.GridViewParameters(
                show_result=True,
                show_v=True,
            ),
        )
