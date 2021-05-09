from __future__ import annotations

from mdp import common
from mdp.scenarios.jacks.scenario.scenario import Scenario
from mdp.scenarios.jacks.scenario.comparison import Comparison


class JacksPolicyIterationV(Scenario):
    def _create_comparison(self) -> Comparison:
        comparison_settings = self._comparison_settings
        comparison_settings.display_every_step = False

        graph3d_values = self._graph3d_values
        graph3d_values.show_graph = False

        grid_view_parameters = self._grid_view_parameters
        grid_view_parameters.show_result = False

        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=comparison_settings,
            settings_list=[
                common.Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.POLICY_ITERATION_DP_V,
                        verbose=False
                    )
                ),
            ],
            graph3d_values=graph3d_values,
            grid_view_parameters=grid_view_parameters,
        )

        # return Comparison(
        #     environment_parameters=self._environment_parameters,
        #     comparison_settings=self._comparison_settings,
        #     settings_list=[
        #         common.Settings(
        #             algorithm_parameters=common.AlgorithmParameters(
        #                 algorithm_type=common.AlgorithmType.POLICY_ITERATION_DP_V,
        #                 verbose=True
        #             )
        #         ),
        #     ],
        #     graph3d_values=self._graph3d_values,
        #     grid_view_parameters=self._grid_view_parameters,
        # )
