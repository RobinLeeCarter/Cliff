from __future__ import annotations

from mdp import common
from mdp.scenario.gambler.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.gambler.comparison.comparison import Comparison
# from mdp.scenarios.gambler.model.environment_parameters import EnvironmentParameters


class GamblerValueIterationV(ComparisonBuilder):
    def create(self):
        return Comparison(
            # environment_parameters=self._environment_parameters,
            comparison_settings=common.Settings(
                gamma=1.0,      # 0.99999
                policy_parameters=common.PolicyParameters(
                    policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
                ),
                algorithm_parameters=common.AlgorithmParameters(
                    theta=0.00001   # accuracy of policy_evaluation
                ),
                display_every_step=False,
            ),
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.DP_VALUE_ITERATION_V,
                        verbose=True
                    )
                ),
            ],
            graph_values=common.GraphValues(
                show_graph=True,
                has_grid=True,
                has_legend=False,
            ),
        )
