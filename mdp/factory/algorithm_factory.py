from __future__ import annotations
from typing import Type, TYPE_CHECKING

from mdp import common
if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm

# tabular
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

# non-tabular
from mdp.model.non_tabular.algorithm.episodic.episodic_sarsa import EpisodicSarsa
from mdp.model.non_tabular.algorithm.episodic.episodic_sarsa_serial_batch import EpisodicSarsaSerialBatch
from mdp.model.non_tabular.algorithm.episodic.episodic_sarsa_parallel_w import EpisodicSarsaParallelW
from mdp.model.non_tabular.algorithm.episodic.episodic_sarsa_parallel_episodes import EpisodicSarsaParallelEpisodes


class AlgorithmFactory:
    def __init__(self, agent: BaseAgent):
        self._agent: BaseAgent = agent

    def create(self, algorithm_parameters: common.AlgorithmParameters) -> BaseAlgorithm:
        algorithm_type: common.AlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[BaseAlgorithm] = BaseAlgorithm.type_registry[algorithm_type]
        algorithm: BaseAlgorithm = type_of_algorithm(self._agent, algorithm_parameters)
        return algorithm

    def get_algorithm_name(self, algorithm_type: common.AlgorithmType) -> str:
        return BaseAlgorithm.name_registry[algorithm_type]

    def get_algorithm_title(self, algorithm_parameters: common.AlgorithmParameters) -> str:
        algorithm_type: common.AlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[BaseAlgorithm] = BaseAlgorithm.type_registry[algorithm_type]
        name: str = BaseAlgorithm.name_registry[algorithm_type]
        return type_of_algorithm.get_title(name, algorithm_parameters)


def __dummy():
    """Stops Pycharm objecting to imports. The imports are needed to generate the registry."""
    return [
        # tabular
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

        # non-tabular
        EpisodicSarsa,
        EpisodicSarsaSerialBatch,
        EpisodicSarsaParallelW,
        EpisodicSarsaParallelEpisodes,
    ]
