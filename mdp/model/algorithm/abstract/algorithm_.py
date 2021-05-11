from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import abc

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
    from mdp.model.policy.policy import Policy
    from mdp import common
from mdp.model.algorithm.value_function.state_function import StateFunction
from mdp.model.algorithm.value_function.state_action_function import StateActionFunction


class Algorithm(abc.ABC):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        self._environment: Environment = environment_
        self._agent: Agent = agent_
        self._algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self._policy_parameters: common.PolicyParameters = policy_parameters
        self._verbose = self._algorithm_parameters.verbose

        self._algorithm_type: Optional[common.AlgorithmType] = None
        self.name: str = "Error: Untitled"
        self.title: str = "Error: Untitled"

        self._gamma: float = self._agent.gamma
        self.V: Optional[StateFunction] = None
        self.Q: Optional[StateActionFunction] = None

    def _create_v(self):
        self.V = StateFunction(self._environment, self._algorithm_parameters.initial_v_value)

    def _create_q(self):
        self.Q = StateActionFunction(self._environment, self._algorithm_parameters.initial_q_value)

    def initialize(self):
        if self.V:
            self.V.initialize_values()
        if self.Q:
            self.Q.initialize_values()

    def parameter_changes(self, iteration: int):
        pass

    def _make_policy_greedy_wrt_q(self):
        # easier and probably faster to include terminal states
        new_policy_vector = np.argmax(self.Q.matrix, axis=1)
        self._agent.target_policy.set_policy_vector(new_policy_vector)

        # for s in range(len(self._environment.states)):
        #     if not self._environment.is_terminal[s]:
        #         # works for single policy or dual policies
        #         self._agent.target_policy[s] = self.Q.argmax_over_actions(s)

    def print_q_coverage_statistics(self):
        self.Q.print_coverage_statistics()

    def __repr__(self):
        return f"{self.title}"

    def derive_v_from_q(self, policy: Optional[Policy] = None):
        if not policy:
            policy = self._agent.policy

        if not self.V:
            self._create_v()

        policy_matrix = policy.get_policy_matrix()
        q_matrix = self.Q.matrix

        expected_v = np.sum( np.dot(policy_matrix, q_matrix.T), axis=1 )    # or something...

        for state in self._environment.states:
            # Sum_over_a( π(a|s).Q(s,a) )
            expected_v: float = 0.0
            for action in self._environment.actions_for_state[state]:
                # π(a|s)
                policy_probability = policy.get_probability(state, action)
                # π(a|s).Q(s,a)
                expected_v += policy_probability * self.Q[state, action]
            self.V[state] = expected_v
