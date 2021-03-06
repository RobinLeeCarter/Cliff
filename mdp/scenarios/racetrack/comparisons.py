from __future__ import annotations

from mdp import common
from mdp.scenarios.racetrack import comparison, environment_parameters, grids


def racetrack_episode() -> comparison.Comparison:
    comparison_ = comparison.Comparison(
        environment_parameters=environment_parameters.EnvironmentParameters(
            grid=grids.TRACK_3,
            extra_reward_for_failure=-100.0,  # 0.0 in problem statement
        ),
        comparison_settings=common.Settings(
            runs=1,
            training_episodes=100_000,
            episode_print_frequency=1000,
            # display_every_step=True,
            dual_policy_relationship=common.DualPolicyRelationship.LINKED_POLICIES
        ),
        breakdown_parameters=common.BreakdownParameters(
            breakdown_type=common.BreakdownType.RETURN_BY_EPISODE
        ),
        settings_list=[
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.EXPECTED_SARSA,
            #     alpha=0.9
            # )),
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.VQ,
            #     alpha=0.2
            # )),
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.Q_LEARNING,
            #     alpha=0.5
            # )),
            common.Settings(algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.OFF_POLICY_MC_CONTROL,
                initial_q_value=-40.0,
            )),
        ],
        graph_values=common.GraphValues(
            moving_average_window_size=101,
            y_min=-200,
            y_max=0
        ),
        grid_view_parameters=common.GridViewParameters(
            grid_view_type=common.GridViewType.POSITION,
            show_values=False,
            show_trail=True
        )
    )
    return comparison_
