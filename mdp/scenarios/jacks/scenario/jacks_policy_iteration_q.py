from __future__ import annotations

from mdp import common
from mdp.scenarios.jacks.scenario.scenario import Scenario
from mdp.scenarios.jacks.scenario.comparison import Comparison


class JacksPolicyIterationQ(Scenario):
    def _create_comparison(self) -> Comparison:
        comparison_settings = self._comparison_settings
        comparison_settings.display_every_step = False

        graph3d_values = self._graph3d_values
        graph3d_values.show_graph = True

        grid_view_parameters = self._grid_view_parameters
        grid_view_parameters.show_result = True

        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=self._comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.DP_POLICY_ITERATION_Q,
                        verbose=True,
                        derive_v_from_q_as_final_step=True,
                    ),
                ),
            ],
            graph3d_values=self._graph3d_values,
            grid_view_parameters=self._grid_view_parameters,
        )
