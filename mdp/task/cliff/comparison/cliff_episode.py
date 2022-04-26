from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.task.cliff.comparison.comparison_builder import ComparisonBuilder
from mdp.task.cliff.comparison.comparison import Comparison
from mdp.task.cliff.model.environment_parameters import EnvironmentParameters


@dataclass
class Settings(common.Settings):
    runs: int = 50
    runs_multiprocessing: common.ParallelContextType = common.ParallelContextType.FORK
    training_episodes: int = 500
    # display_every_step: bool = True


class CliffEpisode(ComparisonBuilder,
                   comparison_type=common.ComparisonType.CLIFF_EPISODE):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=Settings(),
            breakdown_parameters=common.BreakdownParameters(
                breakdown_type=common.BreakdownType.RETURN_BY_EPISODE
            ),
            settings_list=[
                # Settings(algorithm_parameters=common.AlgorithmParameters(
                #     algorithm_type=common.AlgorithmType.EXPECTED_SARSA,
                #     alpha=0.9
                # )),
                # Settings(algorithm_parameters=common.AlgorithmParameters(
                #     algorithm_type=common.AlgorithmType.VQ,
                #     alpha=0.2
                # )),
                Settings(algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.TABULAR_Q_LEARNING,
                    alpha=0.5
                )),
                Settings(algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.TABULAR_SARSA,
                    alpha=0.5
                )),
            ],
            # settings_list_multiprocessing=common.ParallelContextType.SPAWN,
            graph2d_values=common.Graph2DValues(
                has_grid=True,
                has_legend=True,
                moving_average_window_size=19,
                y_min=-100,
                y_max=0,
            ),
            grid_view_parameters=common.GridViewParameters(
                show_demo=True,
                show_q=True
            )
        )
