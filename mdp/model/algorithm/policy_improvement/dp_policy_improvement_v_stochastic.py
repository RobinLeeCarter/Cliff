from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class DpPolicyImprovementVStochastic(DynamicProgrammingV):
    def __init__(self,
                 environment_: Environment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.DP_POLICY_IMPROVEMENT_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name}"

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        policy_stable: bool = False
        while not policy_stable:
            policy_stable = self._policy_improvement(do_call_back)

    def _policy_improvement(self, do_call_back: bool = False) -> bool:
        # TODO: check this is required and not just the same as deterministic
        # assert isinstance(policy_, policy.Deterministic)
        # policy_: policy.Deterministic

        if self._verbose:
            print(f"Starting Policy Improvement ...")

        # policy_matrix[s, a] = π(a|s)
        policy_matrix: np.ndarray = self._agent.policy.get_probability_matrix()
        old_policy_vector: np.ndarray = policy_matrix.argmax(axis=1)
        # state_transition_probabilities[s, a, s'] = p(s'|s,a)
        state_transition_probabilities: np.ndarray = self._environment.dynamics.state_transition_probabilities
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward_np
        # V[s]
        v: np.ndarray = self.V.vector
        gamma = self._agent.gamma

        # expected_return[s,a] = ( Σs',r p(s',r|s,a).r ) + ( γ . Σs' p(s'|s,a).v(s') )
        expected_return: np.ndarray = expected_reward + gamma * np.dot(state_transition_probabilities, v)

        # argmax(a) Σs',r p(s',r|s,a).(r + γ.v(s'))
        new_policy_vector: np.ndarray = expected_return.argmax(axis=1)
        self._agent.policy.set_policy_vector(new_policy_vector)

        policy_stable: bool = np.allclose(old_policy_vector, new_policy_vector)

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")
        if do_call_back:
            self._step_callback()

        return policy_stable
