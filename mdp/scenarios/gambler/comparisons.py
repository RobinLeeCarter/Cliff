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
            gamma=1.0,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
            ),
            algorithm_parameters=common.AlgorithmParameters(
                theta=0.1   # accuracy of policy_evaluation
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
    )
    return comparison


def blackjack_evaluation_q() -> Comparison:
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
            multi_parameter=[False, True]
        ),
        grid_view_parameters=common.GridViewParameters(
            grid_view_type=common.GridViewType.BLACKJACK,
            show_result=True,
            show_policy=True,
        ),
    )
    return comparison


def blackjack_control_es() -> Comparison:
    comparison = Comparison(
        environment_parameters=EnvironmentParameters(
            environment_type=common.EnvironmentType.BLACKJACK,
        ),
        comparison_settings=common.Settings(
            gamma=1.0,
            runs=1,
            training_episodes=100_000,
            episode_print_frequency=10_000,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.DETERMINISTIC,
            ),
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.ON_POLICY_MC_CONTROL,
                    first_visit=True,
                    exploring_starts=True,
                    verbose=True,
                ),
                derive_v_from_q_as_final_step=True
            ),
        ],
        graph3d_values=common.Graph3DValues(
            show_graph=True,
            x_label="Player sum",
            y_label="Dealer showing",
            z_label="V(s)",
            x_min=11,
            x_max=21,
            y_min=1,
            y_max=10,
            z_min=-1.0,
            z_max=1.0,
            multi_parameter=[False, True],
        ),
        grid_view_parameters=common.GridViewParameters(
            grid_view_type=common.GridViewType.BLACKJACK,
            show_result=True,
            show_policy=True,
            multi_parameter=[False, True],
        ),
    )
    return comparison
