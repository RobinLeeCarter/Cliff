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
            training_episodes=500_000,
            episode_print_frequency=10_000,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
            ),
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.MC_PREDICTION_V,
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
            show_graph=True,
            x_label="Player sum",
            y_label="Dealer showing",
            z_label="V(s)",
            x_min=12,
            x_max=21,
            y_min=1,
            y_max=10,
            z_min=-1.0,
            z_max=1.0,
            multi_graph_parameter=[False, True]
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=False,
        ),
    )
    return comparison


def blackjack_comparison_q() -> Comparison:
    comparison = Comparison(
        environment_parameters=EnvironmentParameters(
            environment_type=common.EnvironmentType.BLACKJACK,
        ),
        comparison_settings=common.Settings(
            gamma=1.0,
            runs=1,
            training_episodes=500_000,
            episode_print_frequency=10_000,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
            ),
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.MC_PREDICTION_Q,
                    first_visit=True,
                    verbose=True
                ),
                derive_v_from_q_as_final_step=True
            ),
        ],
        graph_values=common.GraphValues(
            show_graph=False,
            # y_min=0.0,
            # y_max=0.25
        ),
        graph3d_values=common.Graph3DValues(
            show_graph=True,
            x_label="Player sum",
            y_label="Dealer showing",
            z_label="V(s)",
            x_min=12,
            x_max=21,
            y_min=1,
            y_max=10,
            z_min=-1.0,
            z_max=1.0,
            multi_graph_parameter=[False, True]
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=False,
        ),
    )
    return comparison
