from __future__ import annotations

from mdp import common
from mdp.scenarios.gambler.scenario.scenario import Scenario
from mdp.scenarios.gambler.scenario.comparison import Comparison
# from mdp.scenarios.gambler.model.environment_parameters import EnvironmentParameters


class GamblerValueIterationV(Scenario):
    def _create_comparison(self):
        return Comparison(
            # environment_parameters=self._environment_parameters,
            comparison_settings=common.Settings(
                gamma=1.0,      # 0.99999
                policy_parameters=common.PolicyParameters(
                    policy_type=common.PolicyType.DETERMINISTIC,
                ),
                algorithm_parameters=common.AlgorithmParameters(
                    theta=0.00001   # accuracy of policy_evaluation
                ),
                display_every_step=False
            ),
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.VALUE_ITERATION_DP_V_NP,
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
