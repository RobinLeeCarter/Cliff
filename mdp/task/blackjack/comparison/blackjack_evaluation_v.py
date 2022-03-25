from __future__ import annotations

from mdp import common
from mdp.task.blackjack.comparison.comparison_builder import ComparisonBuilder
from mdp.task.blackjack.comparison.comparison import Comparison
from mdp.task.blackjack.model.environment_parameters import EnvironmentParameters
from mdp.task.blackjack.comparison.settings import Settings


class BlackjackEvaluationV(ComparisonBuilder):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=Settings(),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_PREDICTION_V,
                        first_visit=True,
                        verbose=True
                    ),
                    training_episodes=50_000,
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
