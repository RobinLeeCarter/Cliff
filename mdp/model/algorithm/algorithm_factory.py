from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.algorithm import Algorithm as BaseAlgorithm

# from unused.object_style_algorithms.unused_policy_evaluation_dp_q import PolicyEvaluationDpQ
from mdp.model.algorithm.policy_evaluation.dp_policy_evaluation_v_deterministic import DpPolicyEvaluationVDeterministic
from mdp.model.algorithm.policy_evaluation.dp_policy_evaluation_v_stochastic import DpPolicyEvaluationVStochastic
from mdp.model.algorithm.policy_evaluation.dp_policy_evaluation_q_deterministic import DpPolicyEvaluationQDeterministic
from mdp.model.algorithm.policy_evaluation.dp_policy_evaluation_q_stochastic import DpPolicyEvaluationQStochastic
from mdp.model.algorithm.policy_evaluation.constant_alpha_mc import ConstantAlphaMC
from mdp.model.algorithm.policy_evaluation.td_0 import TD0
from mdp.model.algorithm.policy_evaluation.mc_prediction_q import MCPredictionQ
from mdp.model.algorithm.policy_evaluation.mc_prediction_v import MCPredictionV

from mdp.model.algorithm.control.mc_control_on_policy import McControlOnPolicy
from mdp.model.algorithm.control.mc_control_off_policy import McControlOffPolicy
from mdp.model.algorithm.control.vq import VQ
from mdp.model.algorithm.control.expected_sarsa import ExpectedSarsa
from mdp.model.algorithm.control.sarsa import Sarsa
from mdp.model.algorithm.control.q_learning import QLearning


# from unused.object_style_algorithms.unused_policy_improvement_dp_q import PolicyImprovementDpQ
from mdp.model.algorithm.policy_improvement.dp_policy_improvement_v import DpPolicyImprovementV

# from unused.object_style_algorithms.unused_policy_iteration_dp_q import PolicyIterationDpQ
from mdp.model.algorithm.control.dp_policy_iteration_v import DpPolicyIterationV
from mdp.model.algorithm.control.dp_value_iteration_v_deterministic import DpValueIterationV


def algorithm_factory(environment_: Environment,
                      agent_: Agent,
                      algorithm_parameters: common.Settings.algorithm_parameters,
                      policy_parameters: common.PolicyParameters
                      ) -> BaseAlgorithm:
    a = common.AlgorithmType
    algorithm_lookup: dict[a, Type[BaseAlgorithm]] = {
        a.DP_POLICY_EVALUATION_Q_DETERMINISTIC: DpPolicyEvaluationQDeterministic,
        a.DP_POLICY_EVALUATION_Q_STOCHASTIC: DpPolicyEvaluationQStochastic,
        a.DP_POLICY_EVALUATION_V_DETERMINISTIC: DpPolicyEvaluationVDeterministic,
        a.DP_POLICY_EVALUATION_V_STOCHASTIC: DpPolicyEvaluationVStochastic,
        a.DP_POLICY_IMPROVEMENT_V: DpPolicyImprovementV,
        a.DP_POLICY_ITERATION_V: DpPolicyIterationV,
        a.DP_VALUE_ITERATION_V: DpValueIterationV,

        # a.POLICY_EVALUATION_DP_Q: PolicyEvaluationDpQ,
        # a.POLICY_IMPROVEMENT_DP_Q: PolicyImprovementDpQ,
        # a.POLICY_ITERATION_DP_Q: PolicyIterationDpQ,

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
    type_for_algorithm = algorithm_lookup[algorithm_parameters.algorithm_type]
    algorithm_ = type_for_algorithm(environment_, agent_, algorithm_parameters, policy_parameters)
    return algorithm_
