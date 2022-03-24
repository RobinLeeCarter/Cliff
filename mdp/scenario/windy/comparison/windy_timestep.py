from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.scenario.windy.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.windy.comparison.comparison import Comparison
from mdp.scenario.windy.comparison.environment_parameters import EnvironmentParameters


@dataclass
class Settings(common.Settings):
    runs: int = 1
    training_episodes: int = 170
    review_every_step: bool = True
    # display_every_step: bool = True


class WindyTimestep(ComparisonBuilder):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(
                random_wind=self._random_wind,
            ),
            comparison_settings=Settings(),
            breakdown_parameters=common.BreakdownParameters(
                breakdown_type=common.BreakdownType.EPISODE_BY_TIMESTEP,
            ),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.TABULAR_SARSA,
                        alpha=0.5,
                        initial_q_value=0.0,
                    )
                )
            ],
            graph2d_values=common.Graph2DValues(
                has_grid=True,
                has_legend=True,
            ),
            grid_view_parameters=common.GridViewParameters(
                show_demo=True,
                show_q=True,
            )
        )
