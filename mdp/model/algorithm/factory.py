from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.algorithm_ import Algorithm as BaseAlgorithm

from mdp.model.algorithm.policy_evaluation.policy_evaluation_dp_q import PolicyEvaluationDpQ
from mdp.model.algorithm.policy_evaluation.policy_evaluation_dp_v import PolicyEvaluationDpV
from mdp.model.algorithm.policy_evaluation.constant_alpha_mc import ConstantAlphaMC
from mdp.model.algorithm.policy_evaluation.td_0 import TD0
from mdp.model.algorithm.policy_evaluation.mc_prediction_q import MCPredictionQ
from mdp.model.algorithm.policy_evaluation.mc_prediction_v import MCPredictionV

from mdp.model.algorithm.policy_improvement.policy_improvement_dp_q import PolicyImprovementDpQ
from mdp.model.algorithm.policy_improvement.policy_improvement_dp_v import PolicyImprovementDpV

from mdp.model.algorithm.control.on_policy_mc_control import OnPolicyMcControl
from mdp.model.algorithm.control.off_policy_mc_control import OffPolicyMcControl
from mdp.model.algorithm.control.vq import VQ
from mdp.model.algorithm.control.expected_sarsa import ExpectedSarsa
from mdp.model.algorithm.control.sarsa import Sarsa
from mdp.model.algorithm.control.q_learning import QLearning
from mdp.model.algorithm.control.policy_iteration_dp_q import PolicyIterationDpQ
from mdp.model.algorithm.control.policy_iteration_dp_v import PolicyIterationDpV
from mdp.model.algorithm.control.value_iteration_dp_v import ValueIterationDpV


def algorithm_factory(environment_: Environment,
                      agent_: Agent,
                      algorithm_parameters: common.Settings.algorithm_parameters,
                      policy_parameters: common.PolicyParameters
                      ) -> BaseAlgorithm:
    a = common.AlgorithmType
    algorithm_lookup: dict[a, Type[BaseAlgorithm]] = {
        a.POLICY_EVALUATION_DP_V: PolicyEvaluationDpV,
        a.POLICY_IMPROVEMENT_DP_V: PolicyImprovementDpV,
        a.POLICY_ITERATION_DP_V: PolicyIterationDpV,
        a.VALUE_ITERATION_DP_V: ValueIterationDpV,

        a.POLICY_EVALUATION_DP_Q: PolicyEvaluationDpQ,
        a.POLICY_IMPROVEMENT_DP_Q: PolicyImprovementDpQ,
        a.POLICY_ITERATION_DP_Q: PolicyIterationDpQ,

        a.MC_PREDICTION_V: MCPredictionV,
        a.MC_PREDICTION_Q: MCPredictionQ,
        a.ON_POLICY_MC_CONTROL: OnPolicyMcControl,

        a.CONSTANT_ALPHA_MC: ConstantAlphaMC,
        a.TD_0: TD0,
        a.OFF_POLICY_MC_CONTROL: OffPolicyMcControl,

        a.EXPECTED_SARSA: ExpectedSarsa,
        a.Q_LEARNING: QLearning,
        a.SARSA: Sarsa,
        a.VQ: VQ,
    }
    type_for_algorithm = algorithm_lookup[algorithm_parameters.algorithm_type]
    algorithm_ = type_for_algorithm(environment_, agent_, algorithm_parameters, policy_parameters)
    return algorithm_
