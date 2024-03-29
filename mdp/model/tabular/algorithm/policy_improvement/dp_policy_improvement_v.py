from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class DpPolicyImprovementV(DynamicProgrammingV):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        policy_stable: bool = False
        while not policy_stable:
            policy_stable = self._policy_improvement(do_call_back)

    # @profile
    def _policy_improvement(self, do_call_back: bool = False) -> bool:
        # assert isinstance(policy_, policy.Deterministic)
        # policy_: policy.Deterministic

        if self._verbose:
            print(f"Starting Policy Improvement ...")

        # policy_vector[s] = π(s)
        policy_vector: np.ndarray = self._target_policy.get_policy_vector()

        old_policy_vector: np.ndarray = policy_vector.copy()
        # state_transition_probabilities[s, a, s'] = p(s'|s,a)
        state_transition_probabilities: np.ndarray = self._environment.dynamics.state_transition_probabilities
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        expected_reward: np.ndarray = self._environment.dynamics.expected_reward
        # V[s]
        v: np.ndarray = self.V.vector
        gamma = self._agent.gamma

        # expected_return[s,a] = ( Σs',r p(s',r|s,a).r ) + ( γ . Σs' p(s'|s,a).v(s') )
        expected_return: np.ndarray = expected_reward + gamma * np.dot(state_transition_probabilities, v)
        # state_transition_probabilities is a 3D array so numba does not support dot as of 2021-05-11
        # https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#functions

        # argmax(a) Σs',r p(s',r|s,a).(r + γ.v(s'))
        new_policy_vector: np.ndarray = expected_return.argmax(axis=1)
        self._target_policy.set_policy_vector(new_policy_vector)

        policy_stable: bool = np.array_equal(old_policy_vector, new_policy_vector)

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")
        if do_call_back:
            self._step_callback()

        return policy_stable
