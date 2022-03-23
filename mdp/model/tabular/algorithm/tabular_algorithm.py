from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy
    from mdp import common
from mdp.model.tabular.algorithm import linear_algebra as la
from mdp.model.tabular.value_function.state_function import StateFunction
from mdp.model.tabular.value_function.state_action_function import StateActionFunction

from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm


class TabularAlgorithm(BaseAlgorithm, ABC):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._agent: TabularAgent = agent
        self._environment: TabularEnvironment = self._agent.environment
        self._target_policy: Optional[TabularPolicy] = None
        self._behaviour_policy: Optional[TabularPolicy] = None     # if on-policy = self._policy

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

    # @profile
    def _update_target_policy(self, s: int, a: int):
        if self._dual_policy_relationship == common.DualPolicyRelationship.LINKED_POLICIES:
            self._behaviour_policy[s] = a   # this will also update the target policy since linked
        else:
            # in either possible case here we want to update the target policy
            self._target_policy[s] = a

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
            policy = self._target_policy

        # π(a|s)
        policy_matrix = policy.get_probability_matrix()
        # Q(s,a)
        q = self.Q.matrix
        # Sum_over_a( π(a|s).Q(s,a) )
        self.V.vector = la.derive_v_from_q(policy_matrix, q)

    @property
    def target_policy(self) -> Optional[TabularPolicy]:
        return self._target_policy

    @property
    def behaviour_policy(self) -> Optional[TabularPolicy]:
        return self._behaviour_policy

    def apply_result(self, result: common.Result):
        if result.policy_vector:
            self._target_policy.set_policy_vector(result.policy_vector)
        if result.v_vector and self.V:
            self.V.vector = result.v_vector
        if result.q_matrix and self.Q:
            self.Q.set_matrix(result.q_matrix)
