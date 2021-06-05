from __future__ import annotations

from mdp import common
from mdp.scenarios.cliff.scenario.scenario import Scenario
from mdp.scenarios.cliff.scenario.comparison import Comparison


class CliffAlphaEnd(Scenario):
    def _create_comparison(self) -> Comparison:
        # TODO: Make work, once multiprocessing
        return Comparison(
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
            settings_list_multiprocessing=common.ParallelContextType.SPAWN,
            graph_values=common.GraphValues(
                show_graph=True,
                has_grid=True,
                has_legend=True,
                y_min=-140,
                y_max=0,
            ),
        )
