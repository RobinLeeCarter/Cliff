from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import abc

from numba import njit, prange
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
        # TODO: rename agent_ to agent
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

    def _set_policy_greedy_wrt_q(self):
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

    def __repr__(self):
        return f"{self.title}"

    def derive_v_from_q(self, policy: Optional[Policy] = None):
        if not policy:
            policy = self._agent.policy

        if not self.V:
            self._create_v()

        # π(a|s)
        policy_matrix = policy.get_probability_matrix()
        # Q(s,a)
        q_matrix = self.Q.matrix
        # Sum_over_a( π(a|s).Q(s,a) )
        self.V.vector = expected_q(policy_matrix, q_matrix)

        # 30% slower version on 8-core machine
        # self.V.vector = np.einsum('ij,ij->i', policy_matrix, q_matrix)
        # 3x slower version
        # self.V.vector = np.sum(policy_matrix * q_matrix, axis=1)
        # Much slower version!
        # for state in self._environment.states:
        #     # Sum_over_a( π(a|s).Q(s,a) )
        #     expected_v: float = 0.0
        #     for action in self._environment.actions_for_state[state]:
        #         # π(a|s)
        #         policy_probability = policy.get_probability(state, action)
        #         # π(a|s).Q(s,a)
        #         expected_v += policy_probability * self.Q[state, action]
        #     self.V[state] = expected_v


@njit(cache=True, parallel=True)
def expected_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Sum_over_a( π(a|s).Q(s,a) )
    :returns np.einsum('ij,ij->i', policy_matrix, q_matrix)
    """
    out = np.zeros(shape=p.shape[0], dtype=np.float64)
    for i in prange(p.shape[0]):
        for j in range(p.shape[1]):
            out[i] += p[i, j] * q[i, j]
    return out
