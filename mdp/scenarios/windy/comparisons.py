from __future__ import annotations

from mdp import common


def windy_timestep(random_wind: bool = False) -> common.Comparison:
    comparison = common.Comparison(
        environment_parameters=common.EnvironmentParameters(
            environment_type=common.EnvironmentType.WINDY,
            actions_list=common.ActionsList.FOUR_MOVES,
            random_wind=random_wind,
        ),
        comparison_settings=common.Settings(
            runs=1,
            training_episodes=170,
            review_every_step=True,
            # display_every_step=True,
        ),
        breakdown_parameters=common.BreakdownParameters(
            breakdown_type=common.BreakdownType.EPISODE_BY_TIMESTEP,
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.SARSA,
                    alpha=0.5,
                    initial_q_value=0.0,
                )
            )
        ],
        # graph_values=common.GraphValues(
        #     show_graph=False
        # ),
        # grid_view_parameters=common.GridViewParameters(
        #     show_demo=False
        # )
    )
    return comparison
