from __future__ import annotations

from mdp import common
from mdp.task.jacks.comparison.comparison_builder import ComparisonBuilder
from mdp.task.jacks.comparison.comparison import Comparison
from mdp.task.jacks.comparison.settings import Settings


class JacksPolicyIterationV(ComparisonBuilder,
                            comparison_type=common.ComparisonType.JACKS_POLICY_ITERATION_V):
    def create(self) -> Comparison:
        graph3d_values = self._graph3d_values
        grid_view_parameters = self._grid_view_parameters

        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.DP_POLICY_ITERATION_V,
                        verbose=False,
                        theta=0.1  # accuracy of policy_evaluation
                    )
                ),
            ],
            graph3d_values=graph3d_values,
            grid_view_parameters=grid_view_parameters,
        )
