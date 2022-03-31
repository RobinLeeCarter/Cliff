from __future__ import annotations

from mdp import common
from mdp.task.blackjack.comparison.comparison_builder import ComparisonBuilder
from mdp.task.blackjack.comparison.comparison import Comparison
from mdp.task.blackjack.model.environment_parameters import EnvironmentParameters
from mdp.task.blackjack.comparison.settings import Settings


class BlackjackControlES(ComparisonBuilder,
                         comparison_type=common.ComparisonType.BLACKJACK_CONTROL_ES):
    def create(self) -> Comparison:
        # self._comparison_settings.training_episodes = 100_000
        comparison = Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=Settings(),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.MC_CONTROL_ON_POLICY,
                        first_visit=True,
                        exploring_starts=True,
                        derive_v_from_q_as_final_step=True,
                        verbose=True,
                    ),
                    training_episodes=100_000,
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
        return comparison
