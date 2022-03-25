from __future__ import annotations

from mdp import common
from mdp.scenario.jacks.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.jacks.comparison.comparison import Comparison
from mdp.scenario.jacks.comparison.settings import Settings


class JacksValueIterationQ(ComparisonBuilder):
    def create(self) -> Comparison:
        graph3d_values = self._graph3d_values

        grid_view_parameters = self._grid_view_parameters

        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        algorithm_type=common.AlgorithmType.DP_VALUE_ITERATION_Q,
                        verbose=True,
                        derive_v_from_q_as_final_step=True,
                        theta=0.1  # accuracy of policy_evaluation
                    ),
                ),
            ],
            graph3d_values=graph3d_values,
            grid_view_parameters=grid_view_parameters,
        )
