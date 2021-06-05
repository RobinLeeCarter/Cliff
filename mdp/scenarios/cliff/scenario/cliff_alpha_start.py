from __future__ import annotations

from mdp import common
from mdp.scenarios.cliff.scenario.scenario import Scenario
from mdp.scenarios.cliff.scenario.comparison import Comparison


class CliffAlphaStart(Scenario):
    def _create_comparison(self) -> Comparison:
        return Comparison(
            # environment_parameters=environment_parameters.EnvironmentParameters(
            #     actions_list=common.ActionsList.FOUR_MOVES,
            # ),
            comparison_settings=common.Settings(
                runs=10,
                # runs_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
                training_episodes=100,
                # display_every_step=True,
            ),
            breakdown_parameters=common.BreakdownAlgorithmByAlpha(
                breakdown_type=common.BreakdownType.RETURN_BY_ALPHA,
                alpha_min=0.1,
                alpha_max=1.0,
                alpha_step=0.1,
                algorithm_type_list=[
                    common.AlgorithmType.EXPECTED_SARSA,
                    common.AlgorithmType.VQ,
                    common.AlgorithmType.Q_LEARNING,
                    common.AlgorithmType.SARSA
                ],
            ),
            settings_list_multiprocessing=common.ParallelContextType.FORK_GLOBAL,
            graph_values=common.GraphValues(
                show_graph=True,
                has_grid=True,
                has_legend=True,
                y_min=-140,
                y_max=0,
            ),
            grid_view_parameters=common.GridViewParameters(
                show_policy=True,
                show_q=True
            )
        )
