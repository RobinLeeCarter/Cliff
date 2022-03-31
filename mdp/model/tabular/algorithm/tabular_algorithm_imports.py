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


def dummy_list():
    return [
        DpPolicyEvaluationQDeterministic,
        DpPolicyEvaluationQStochastic,
        DpPolicyEvaluationVDeterministic,
        DpPolicyEvaluationVStochastic,

        DpPolicyImprovementQ,
        DpPolicyImprovementV,
        DpPolicyIterationQ,
        DpPolicyIterationV,
        DpValueIterationQ,
        DpValueIterationV,

        MCPredictionV,
        MCPredictionQ,
        McControlOnPolicy,
        ConstantAlphaMC,
        McControlOffPolicy,

        TD0,
        ExpectedSarsa,
        QLearning,
        Sarsa,
        VQ,
    ]
