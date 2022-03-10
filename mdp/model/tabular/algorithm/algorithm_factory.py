from __future__ import annotations
from typing import TYPE_CHECKING, Type  # , TypeVar, Generic

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm
from mdp import common
# from mdp.model.tabular.environment.tabular_state import TabularState
# from mdp.model.tabular.environment.tabular_action import TabularAction

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


# State = TypeVar('State', bound=TabularState)
# Action = TypeVar('Action', bound=TabularAction)


class AlgorithmFactory:     # Generic[State, Action]
    def __init__(self, environment: TabularEnvironment, agent: Agent):      # [State, Action]
        self._environment: TabularEnvironment = environment                 # [State, Action]
        self._agent: Agent = agent

        self._algorithm_lookup: dict[common.TabularAlgorithmType, Type[Algorithm]] = self._get_algorithm_lookup()
        self._name_lookup: dict[common.TabularAlgorithmType, str] = self._get_name_lookup()

    def _get_algorithm_lookup(self) -> dict[common.TabularAlgorithmType, Type[Algorithm]]:
        a = common.TabularAlgorithmType
        return {
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

    def _get_name_lookup(self) -> dict[common.TabularAlgorithmType, str]:
        a = common.TabularAlgorithmType
        return {
            a.DP_POLICY_EVALUATION_V_DETERMINISTIC: 'Policy Evaluation DP (V) Deterministic',
            a.DP_POLICY_EVALUATION_V_STOCHASTIC: 'Policy Evaluation DP (V) Stochastic',
            a.DP_POLICY_IMPROVEMENT_V: 'Policy Improvement DP (V)',
            a.DP_POLICY_ITERATION_V: 'Policy Iteration DP (V)',
            a.DP_VALUE_ITERATION_V: 'Value Iteration DP (V)',

            a.DP_POLICY_EVALUATION_Q_DETERMINISTIC: 'Policy Evaluation DP (Q) Deterministic',
            a.DP_POLICY_EVALUATION_Q_STOCHASTIC: 'Policy Evaluation DP (Q) Stochastic',
            a.DP_POLICY_IMPROVEMENT_Q: 'Policy Improvement DP (Q)',
            a.DP_POLICY_ITERATION_Q: 'Policy Iteration DP (Q)',
            a.DP_VALUE_ITERATION_Q: 'Value Iteration DP (Q)',

            a.MC_PREDICTION_V: 'MC Prediction (V)',
            a.MC_PREDICTION_Q: 'MC Prediction (Q)',
            a.MC_CONTROL_ON_POLICY: 'On-policy MC Control',

            a.MC_CONSTANT_ALPHA: 'Constant-α MC',
            a.MC_CONTROL_OFF_POLICY: 'Off-policy MC Control',

            a.TD_0: 'TD(0)',
            a.EXPECTED_SARSA: 'Expected Sarsa',
            a.Q_LEARNING: 'Q-learning',
            a.SARSA: 'Sarsa',
            a.VQ: 'VQ',
        }

    def create(self, algorithm_parameters: common.Settings.algorithm_parameters) -> Algorithm:
        algorithm_type: common.TabularAlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[Algorithm] = self._algorithm_lookup[algorithm_type]
        algorithm_name: str = self._name_lookup[algorithm_type]

        algorithm: Algorithm = type_of_algorithm(self._environment,
                                                 self._agent,
                                                 algorithm_parameters,
                                                 algorithm_name)
        return algorithm
