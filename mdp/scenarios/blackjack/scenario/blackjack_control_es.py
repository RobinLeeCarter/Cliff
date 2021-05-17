from __future__ import annotations

from mdp import common
from mdp.scenarios.blackjack.scenario.scenario import Scenario
from mdp.scenarios.blackjack.scenario.comparison import Comparison


class BlackjackControlES(Scenario):
    def _create_comparison(self) -> Comparison:
        # self._comparison_settings.training_episodes = 100_000

        comparison = Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=self._comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.ON_POLICY_MC_CONTROL,
                        first_visit=True,
                        exploring_starts=True,
                        verbose=True,
                        theta=0.1
                    ),
                    derive_v_from_q_as_final_step=True,
                    training_episodes=100_000
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
        return comparison
