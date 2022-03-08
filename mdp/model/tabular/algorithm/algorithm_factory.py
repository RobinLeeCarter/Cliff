from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
from mdp import common
from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm

from mdp.model.tabular.algorithm.policy_evaluation.dp_policy_evaluation_v_deterministic\
    import DpPolicyEvaluationVDeterministic
from mdp.model.tabular.algorithm.policy_evaluation.dp_policy_evaluation_v_stochastic \
    import DpPolicyEvaluationVStochastic
from mdp.model.tabular.algorithm.policy_evaluation.dp_policy_evaluation_q_deterministic \
    import DpPolicyEvaluationQDeterministic
from mdp.model.tabular.algorithm.policy_evaluation.dp_policy_evaluation_q_stochastic \
    import DpPolicyEvaluationQStochastic

from mdp.model.tabular.algorithm.policy_improvement.dp_policy_improvement_q import DpPolicyImprovementQ
from mdp.model.tabular.algorithm.policy_improvement.dp_policy_improvement_v import DpPolicyImprovementV
from mdp.model.tabular.algorithm.control.dp_policy_iteration_q import DpPolicyIterationQ
from mdp.model.tabular.algorithm.control.dp_policy_iteration_v import DpPolicyIterationV
from mdp.model.tabular.algorithm.control.dp_value_iteration_q_deterministic import DpValueIterationQ
from mdp.model.tabular.algorithm.control.dp_value_iteration_v_deterministic import DpValueIterationV

from mdp.model.tabular.algorithm.policy_evaluation.constant_alpha_mc import ConstantAlphaMC
from mdp.model.tabular.algorithm.policy_evaluation.td_0 import TD0
from mdp.model.tabular.algorithm.policy_evaluation.mc_prediction_q import MCPredictionQ
from mdp.model.tabular.algorithm.policy_evaluation.mc_prediction_v import MCPredictionV

from mdp.model.tabular.algorithm.control.mc_control_on_policy import McControlOnPolicy
from mdp.model.tabular.algorithm.control.mc_control_off_policy import McControlOffPolicy
from mdp.model.tabular.algorithm.control.vq import VQ
from mdp.model.tabular.algorithm.control.expected_sarsa import ExpectedSarsa
from mdp.model.tabular.algorithm.control.sarsa import Sarsa
from mdp.model.tabular.algorithm.control.q_learning import QLearning

from mdp.model.tabular.algorithm.control.__register import register_control_algorithms


class AlgorithmFactory:
    def __init__(self, environment: TabularEnvironment, agent: Agent):
        self._environment: TabularEnvironment = environment
        self._agent: Agent = agent
        a = common.AlgorithmType
        self._algorithm_lookup: dict[a, Type[Algorithm]] = {}
        register_control_algorithms(self.register)

        self._algorithm_lookup: dict[a, Type[Algorithm]] = {
            a.DP_POLICY_EVALUATION_Q_DETERMINISTIC: DpPolicyEvaluationQDeterministic,
            a.DP_POLICY_EVALUATION_Q_STOCHASTIC: DpPolicyEvaluationQStochastic,
            a.DP_POLICY_EVALUATION_V_DETERMINISTIC: DpPolicyEvaluationVDeterministic,
            a.DP_POLICY_EVALUATION_V_STOCHASTIC: DpPolicyEvaluationVStochastic,
            a.DP_POLICY_IMPROVEMENT_Q: DpPolicyImprovementQ,
            a.DP_POLICY_IMPROVEMENT_V: DpPolicyImprovementV,
            a.DP_POLICY_ITERATION_Q: DpPolicyIterationQ,
            a.DP_POLICY_ITERATION_V: DpPolicyIterationV,
            a.DP_VALUE_ITERATION_Q: DpValueIterationQ,
            a.DP_VALUE_ITERATION_V: DpValueIterationV,

            a.MC_PREDICTION_V: MCPredictionV,
            a.MC_PREDICTION_Q: MCPredictionQ,
            a.MC_CONTROL_ON_POLICY: McControlOnPolicy,
            a.MC_CONSTANT_ALPHA: ConstantAlphaMC,
            a.MC_CONTROL_OFF_POLICY: McControlOffPolicy,

            a.TD_0: TD0,
            a.EXPECTED_SARSA: ExpectedSarsa,
            a.Q_LEARNING: QLearning,
            a.SARSA: Sarsa,
            a.VQ: VQ,
        }

    def register(self, type_for_algorithm: Type[Algorithm]):
        self._algorithm_lookup[type_for_algorithm.algorithm_type] = type_for_algorithm

    def create_algorithm(self, algorithm_parameters: common.Settings.algorithm_parameters) -> Algorithm:
        type_for_algorithm: Type[Algorithm] = self._algorithm_lookup[algorithm_parameters.algorithm_type]
        algorithm: Algorithm = type_for_algorithm(self._environment, self._agent, algorithm_parameters)
        return algorithm
