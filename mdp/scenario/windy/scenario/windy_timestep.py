from __future__ import annotations

from mdp import common
from mdp.scenario.windy.scenario.scenario import Scenario
from mdp.scenario.windy.scenario.comparison import Comparison
from mdp.scenario.windy.model.environment_parameters import EnvironmentParameters


class WindyTimestep(Scenario):
    def _create_comparison(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(
                # actions_list=common.ActionsList.FOUR_MOVES,
                random_wind=self._random_wind,
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
                        algorithm_type=common.TabularAlgorithmType.SARSA,
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
