from __future__ import annotations

from mdp import common
from mdp.scenarios.blackjack.scenario.scenario import Scenario
from mdp.scenarios.blackjack.scenario.comparison import Comparison


class BlackjackEvaluationV(Scenario):
    def _create_comparison(self):
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=self._comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_PREDICTION_V,
                        first_visit=True,
                        verbose=True
                    ),
                    training_episodes=50_000,
                ),
            ],
            settings_list_multiprocessing=common.ParallelContextType.NONE,
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
