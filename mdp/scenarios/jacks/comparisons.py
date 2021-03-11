from __future__ import annotations

from mdp import common
from mdp.scenarios.jacks import comparison


def jacks_policy_evaluation() -> comparison.Comparison:
    comparison_ = comparison.Comparison(
        # environment_parameters=common.EnvironmentParameters(
        #     environment_type=common.EnvironmentType.JACKS
        # ),
        comparison_settings=common.Settings(
            gamma=0.9,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
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
        graph3d_values=common.Graph3DValues(
            show_graph=True,
            x_label="Cars at 1st location",
            y_label="Cars at 2nd location",
            z_label="V(s)",
            x_min=0,
            x_max=20,
            y_min=0,
            y_max=20,
            z_min=400.0,
            z_max=700.0,
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=False
        ),
    )
    return comparison_
