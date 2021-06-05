from __future__ import annotations

from mdp import common
from mdp.scenarios.cliff.scenario.scenario import Scenario
from mdp.scenarios.cliff.scenario.comparison import Comparison


class CliffEpisode(Scenario):
    def _create_comparison(self) -> Comparison:
        return Comparison(
            # environment_parameters=environment_parameters.EnvironmentParameters(
            #     actions_list=common.ActionsList.FOUR_MOVES,
            # ),
            comparison_settings=common.Settings(
                runs=50,
                runs_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
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
            # settings_list_multiprocessing=common.ParallelContextType.SPAWN,
            graph_values=common.GraphValues(
                show_graph=True,
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
