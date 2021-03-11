from __future__ import annotations

from mdp import common
from mdp.scenarios.jacks import comparison


def jacks_policy_evaluation() -> comparison.Comparison:
    comparison_ = comparison.Comparison(
        # environment_parameters=common.EnvironmentParameters(
        #     environment_type=common.EnvironmentType.JACKS
        # ),
        comparison_settings=common.Settings(
            runs=100,
            training_episodes=100,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
                initialize=True,
            ),
            algorithm_parameters=common.AlgorithmParameters(
                theta=0.1   # accuracy of policy_evaluation
            )
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.POLICY_EVALUATION_DP_V,
                    verbose=True
                )
            ),
        ],
        graph_values=common.GraphValues(
            show_graph=False,
            # y_min=0.0,
            # y_max=0.25
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=False
        ),
    )
    return comparison_
