from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.tabular_environment import TabularEnvironment
from mdp.model.policy import policy


class Deterministic(policy.Policy):
    def __init__(self, environment_: TabularEnvironment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        state_count = len(self._environment.states)
        self.policy_vector: np.ndarray = np.zeros(shape=state_count, dtype=int)

    def _get_a(self, s: int) -> int:
        return self.policy_vector[s]

    def __setitem__(self, s: int, a: int):
        if self._store_matrix:
            prev_a = self.policy_vector[s]
            self._policy_matrix[s, prev_a] = 0.0
            self._policy_matrix[s, a] = 1.0
        self.policy_vector[s] = a

    def _calc_probability(self, s: int, a: int) -> float:
        if a == self.policy_vector[s]:
            return 1.0
        else:
            return 0.0

    def _calc_probability_vector(self, s: int) -> np.ndarray:
        action_count = len(self._environment.actions)
        probability_vector = np.zeros(shape=action_count, dtype=float)
        a = self.policy_vector[s]
        probability_vector[a] = 1.0
        return probability_vector

    def _calc_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
        i = np.arange(state_count)
        policy_matrix[i, self.policy_vector] = 1.0
        return policy_matrix

    def get_policy_vector(self) -> np.ndarray:
        return self.policy_vector

    def set_policy_vector(self, policy_vector: np.ndarray):
        self.policy_vector = policy_vector
        if self._store_matrix:
            self._policy_matrix = self._calc_policy_matrix()
