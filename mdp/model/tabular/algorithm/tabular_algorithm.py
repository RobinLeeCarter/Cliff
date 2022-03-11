from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.agent import Agent
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy
    from mdp import common
from mdp.model.tabular.algorithm import linear_algebra as la
from mdp.model.tabular.value_function.state_function import StateFunction
from mdp.model.tabular.value_function.state_action_function import StateActionFunction

from mdp.model.general.algorithm.general_algorithm import GeneralAlgorithm


class TabularAlgorithm(GeneralAlgorithm, ABC):
    def __init__(self,
                 environment: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(environment, agent, algorithm_parameters, name)
        self._environment: TabularEnvironment = environment
        self._agent: Agent = agent

        self.V: Optional[StateFunction] = None
        self.Q: Optional[StateActionFunction] = None
        if self._algorithm_parameters.derive_v_from_q_as_final_step:
            self._create_v()

    def _create_v(self):
        if not self.V:  # could have been already created in __init__
            self.V = StateFunction(self._environment, self._algorithm_parameters.initial_v_value)

    def _create_q(self):
        self.Q = StateActionFunction(self._environment, self._algorithm_parameters.initial_q_value)

    def initialize(self):
        if self.V:
            self.V.initialize_values()
        if self.Q:
            self.Q.initialize_values()

    def _set_target_policy_greedy_wrt_q(self):
        self._agent.target_policy.set_policy_vector(self.Q.argmax.copy())

        # easier and probably faster to include terminal states
        # new_policy_vector = np.argmax(self.Q.matrix, axis=1)
        # self._agent.target_policy.set_policy_vector(new_policy_vector)

        # for s in range(len(self._environment.states)):
        #     if not self._environment.is_terminal[s]:
        #         # works for single policy or dual policies
        #         self._agent.target_policy[s] = self.Q.argmax_over_actions(s)

    def print_q_coverage_statistics(self):
        self.Q.print_coverage_statistics()

    def derive_v_from_q(self, policy: Optional[TabularPolicy] = None):
        if not policy:
            policy = self._agent.policy

        # π(a|s)
        policy_matrix = policy.get_probability_matrix()
        # Q(s,a)
        q = self.Q.matrix
        # Sum_over_a( π(a|s).Q(s,a) )
        self.V.vector = la.derive_v_from_q(policy_matrix, q)
