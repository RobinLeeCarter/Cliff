from __future__ import annotations

from mdp import common
from mdp.scenarios.blackjack.comparison import Comparison
from mdp.scenarios.blackjack.environment_parameters import EnvironmentParameters


def blackjack_comparison_v() -> Comparison:
    comparison = Comparison(
        environment_parameters=EnvironmentParameters(
            environment_type=common.EnvironmentType.BLACKJACK,
        ),
        comparison_settings=common.Settings(
            gamma=1.0,
            runs=1,
            training_episodes=100,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
            ),
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.MC_PREDICTION,
                    first_visit=True,
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
            show_graph=False,
            # x_label="Cars at 1st location",
            # y_label="Cars at 2nd location",
            # z_label="V(s)",
            # x_min=0,
            # x_max=_max_cars,
            # y_min=0,
            # y_max=_max_cars,
            # z_min=400.0,
            # z_max=700.0,
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=False,
        ),
    )
    return comparison
