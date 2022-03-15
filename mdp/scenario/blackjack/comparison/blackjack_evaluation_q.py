from __future__ import annotations

from mdp import common
from mdp.scenario.blackjack.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.blackjack.comparison.comparison import Comparison
from mdp.scenario.blackjack.comparison.environment_parameters import EnvironmentParameters


class BlackjackEvaluationQ(ComparisonBuilder):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=self._comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_PREDICTION_Q,
                        first_visit=True,
                        verbose=True,
                        derive_v_from_q_as_final_step=True
                    ),
                    training_episodes=100_000,
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
