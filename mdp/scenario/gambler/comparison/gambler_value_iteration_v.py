from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.scenario.gambler.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.gambler.comparison.comparison import Comparison
from mdp.scenario.gambler.comparison.environment_parameters import EnvironmentParameters


@dataclass
class Settings(common.Settings):
    gamma: float = 1.0  # 0.99999
    policy_parameters: common.PolicyParameters = common.PolicyParameters(
        policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
    )
    display_every_step: bool = False


class GamblerValueIterationV(ComparisonBuilder):
    def create(self):
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=Settings(),
            settings_list=[
                Settings(
                    algorithm_parameters=common.AlgorithmParameters(
                        theta=0.00001,  # accuracy of policy_evaluation
                        algorithm_type=common.AlgorithmType.DP_VALUE_ITERATION_V,
                        verbose=True
                    ),
                ),
            ],
            graph2d_values=common.Graph2DValues(),
        )


# comparison_settings = common.Settings(
#     gamma=1.0,  # 0.99999
#     policy_parameters=common.PolicyParameters(
#         policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
#     ),
#     algorithm_parameters=common.AlgorithmParameters(
#         theta=0.00001  # accuracy of policy_evaluation
#     ),
#     display_every_step=False,
# ),
# settings_list = [
#                     Settings(
#                         gamma=1.0,  # 0.99999
#                         policy_parameters=common.PolicyParameters(
#                             policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
#                         ),
#                         algorithm_parameters=common.AlgorithmParameters(
#                             theta=0.00001,  # accuracy of policy_evaluation
#                             algorithm_type=common.AlgorithmType.DP_VALUE_ITERATION_V,
#                             verbose=True
#                         ),
#                         display_every_step=False,
#                     ),
#                 ],
