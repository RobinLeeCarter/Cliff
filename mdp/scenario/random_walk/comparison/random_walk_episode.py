from __future__ import annotations

from mdp import common
from mdp.scenario.random_walk.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.random_walk.comparison.comparison import Comparison
from mdp.scenario.random_walk.comparison.environment_parameters import EnvironmentParameters


class RandomWalkEpisode(ComparisonBuilder):
    def create(self):
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=common.Settings(
                runs=100,
                # runs_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
                training_episodes=100,
                policy_parameters=common.PolicyParameters(
                    policy_type=common.PolicyType.TABULAR_NONE
                ),
                algorithm_parameters=common.AlgorithmParameters(
                    initial_v_value=0.5
                )
            ),
            breakdown_parameters=common.BreakdownParameters(
                breakdown_type=common.BreakdownType.RMS_BY_EPISODE,
            ),
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_TD_0,
                        alpha=0.05,
                    )
                ),
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_TD_0,
                        alpha=0.1,
                    )
                ),
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_TD_0,
                        alpha=0.15,
                    )
                ),
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.01,
                    )
                ),
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.02,
                    )
                ),
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.03,
                    )
                ),
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONSTANT_ALPHA,
                        alpha=0.04,
                    )
                ),
            ],
            settings_list_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
            graph_values=common.GraphValues(
                show_graph=True,
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
