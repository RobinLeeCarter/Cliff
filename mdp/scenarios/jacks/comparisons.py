from __future__ import annotations

from mdp import common
from mdp.scenarios.jacks.comparison import Comparison
from mdp.scenarios.jacks.environment_parameters import EnvironmentParameters

max_cars: int = 10      # problem statement = 20


def jacks_policy_evaluation() -> Comparison:
    comparison = Comparison(
        environment_parameters=EnvironmentParameters(
            environment_type=common.EnvironmentType.JACKS,
            max_cars=max_cars,
            # rental_rate_2=0.0,
            # return_rate_2=0.0,
        ),
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
            x_max=max_cars,
            y_min=0,
            y_max=max_cars,
            # z_min=400.0,
            # z_max=700.0,
        ),
        grid_view_parameters=common.GridViewParameters(
            grid_view_type=common.GridViewType.JACKS,
            show_demo=False,
            show_values=True,
        ),
    )
    return comparison
