from __future__ import annotations

from mdp import common
from mdp.scenarios.gambler.comparison import Comparison
from mdp.scenarios.gambler.environment_parameters import EnvironmentParameters


def gambler_value_iteration_v() -> Comparison:
    comparison = Comparison(
        environment_parameters=EnvironmentParameters(
            environment_type=common.EnvironmentType.GAMBLER,
            probability_heads=0.5,
        ),
        comparison_settings=common.Settings(
            gamma=1.0,      # 0.99999
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
            ),
            algorithm_parameters=common.AlgorithmParameters(
                theta=0.001   # accuracy of policy_evaluation
            ),
            display_every_step=False
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.VALUE_ITERATION_DP_V,
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
    return comparison
