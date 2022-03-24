from __future__ import annotations

from mdp import common
from mdp.scenario.jacks.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.jacks.comparison.comparison import Comparison
from mdp.scenario.jacks.comparison.settings import Settings


class JacksPolicyEvaluationV(ComparisonBuilder):
    def create(self) -> Comparison:
        graph3d_values = self._graph3d_values

        grid_view_parameters = self._grid_view_parameters
        grid_view_parameters.show_result = True

        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.DP_POLICY_EVALUATION_V_DETERMINISTIC,
                        verbose=True,
                        theta=0.1  # accuracy of policy_evaluation
                    )
                ),
            ],
            graph3d_values=graph3d_values,
            grid_view_parameters=grid_view_parameters,
        )
