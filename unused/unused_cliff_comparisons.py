from __future__ import annotations

from mdp import common
from mdp.scenarios.cliff.scenario.comparison import Comparison


def cliff_alpha_start() -> Comparison:
    comparison = Comparison(
        # environment_parameters=environment_parameters.EnvironmentParameters(
        #     actions_list=common.ActionsList.FOUR_MOVES,
        # ),
        comparison_settings=common.Settings(
            runs=10,
            training_episodes=100,
        ),
        breakdown_parameters=common.BreakdownAlgorithmByAlpha(
            breakdown_type=common.BreakdownType.RETURN_BY_ALPHA,
            alpha_min=0.1,
            alpha_max=1.0,
            alpha_step=0.05,
            algorithm_type_list=[
                common.AlgorithmType.EXPECTED_SARSA,
                # common.AlgorithmType.VQ,
                common.AlgorithmType.Q_LEARNING,
                common.AlgorithmType.SARSA
            ],
        ),
        graph_values=common.GraphValues(
            show_graph=True,
            has_grid=True,
            has_legend=True,
            y_min=-140,
            y_max=0,
        ),
    )
    return comparison


def cliff_alpha_end() -> Comparison:
    comparison = Comparison(
        # environment_parameters=environment_parameters.EnvironmentParameters(
        #     actions_list=common.ActionsList.FOUR_MOVES,
        # ),
        comparison_settings=common.Settings(
            runs=1,
            training_episodes=100_000,
        ),
        breakdown_parameters=common.BreakdownAlgorithmByAlpha(
            breakdown_type=common.BreakdownType.RETURN_BY_ALPHA,
            alpha_min=0.1,
            alpha_max=1.0,
            alpha_step=0.05,
            algorithm_type_list=[
                common.AlgorithmType.EXPECTED_SARSA,
                # common.AlgorithmType.VQ,
                common.AlgorithmType.Q_LEARNING,
                common.AlgorithmType.SARSA
            ],
        ),
        graph_values=common.GraphValues(
            show_graph=True,
            has_grid=True,
            has_legend=True,
            y_min=-140,
            y_max=0,
        ),
    )
    return comparison


def cliff_episode() -> Comparison:
    comparison = Comparison(
        # environment_parameters=environment_parameters.EnvironmentParameters(
        #     actions_list=common.ActionsList.FOUR_MOVES,
        # ),
        comparison_settings=common.Settings(
            runs=50,
            training_episodes=500,
            # display_every_step=True,
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
            common.Settings(algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.Q_LEARNING,
                alpha=0.5
            )),
            common.Settings(algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.SARSA,
                alpha=0.5
            )),
        ],
        graph_values=common.GraphValues(
            show_graph=True,
            has_grid=True,
            has_legend=True,
            moving_average_window_size=19,
            y_min=-100,
            y_max=0,
        ),
    )
    return comparison
