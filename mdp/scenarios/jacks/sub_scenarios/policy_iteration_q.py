from __future__ import annotations

from mdp import common
from mdp.scenarios.jacks.scenario import Scenario
from mdp.scenarios.jacks.comparison import Comparison


class PolicyIterationQ(Scenario):
    def _set_comparison(self):
        self._comparison_settings.derive_v_from_q_as_final_step = True

        self._comparison = Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=self._comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.POLICY_ITERATION_DP_Q,
                        verbose=True
                    )
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
