from __future__ import annotations

from mdp import common
from mdp.scenarios.windy import comparison, environment_parameters


def windy_timestep(random_wind: bool = False) -> comparison.Comparison:
    comparison_ = comparison.Comparison(
        environment_parameters=environment_parameters.EnvironmentParameters(
            # actions_list=common.ActionsList.FOUR_MOVES,
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
        graph_values=common.GraphValues(
            show_graph=True,
            has_grid=True,
            has_legend=True,
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=True,
            show_q=True,
        )
    )
    return comparison_
