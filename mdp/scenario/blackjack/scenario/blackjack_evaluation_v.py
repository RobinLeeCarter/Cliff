from __future__ import annotations

from mdp import common
from mdp.scenario.blackjack.scenario.scenario import Scenario
from mdp.scenario.blackjack.scenario.comparison import Comparison


class BlackjackEvaluationV(Scenario):
    def _create_comparison(self) -> Comparison:
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=self._comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.TabularAlgorithmType.MC_PREDICTION_V,
                        first_visit=True,
                        verbose=True
                    ),
                    training_episodes=50_000,
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
