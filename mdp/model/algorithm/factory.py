from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.model import environment, agent
from mdp import common
from mdp.model.algorithm import abstract, control, policy_evaluation, policy_improvement


def factory(environment_: environment.Environment,
            agent_: agent.Agent,
            algorithm_parameters: common.Settings.algorithm_parameters,
            policy_parameters: common.PolicyParameters
            ) -> abstract.Algorithm:
    a = common.AlgorithmType
    algorithm_lookup: dict[a, Type[abstract.Algorithm]] = {
        a.POLICY_EVALUATION_DP_V: policy_evaluation.PolicyEvaluationDpV,
        a.POLICY_IMPROVEMENT_DP_V: policy_improvement.PolicyImprovementDpV,
        a.POLICY_ITERATION_DP_V: control.PolicyIterationDpV,

        a.TD_0: policy_evaluation.TD0,
        a.OFF_POLICY_MC_CONTROL: control.OffPolicyMcControl,

        a.EXPECTED_SARSA: control.ExpectedSarsa,
        a.Q_LEARNING: control.QLearning,
        a.SARSA: control.Sarsa,
        a.VQ: control.VQ,
        a.CONSTANT_ALPHA_MC: policy_evaluation.ConstantAlphaMC,
    }
    type_for_algorithm = algorithm_lookup[algorithm_parameters.algorithm_type]
    algorithm_ = type_for_algorithm(environment_, agent_, algorithm_parameters, policy_parameters)
    return algorithm_
