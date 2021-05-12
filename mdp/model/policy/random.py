from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
from mdp import common
from mdp.model.policy.policy import Policy


class Random(Policy):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        if self._store_matrix:  # and isinstance(self, Random)
            self._policy_matrix = self._calc_policy_matrix()

    def _get_a(self, s: int) -> int:
        return common.rng.choice(
            np.flatnonzero(
                self._environment.s_a_compatibility[s, :]
            )
        )

    def _calc_probability(self, s: int, a: int) -> float:
        if self._environment.s_a_compatibility[s, a]:
            return self._environment.one_over_possible_actions[s]
        else:
            return 0.0

    def _calc_probability_vector(self, s: int) -> np.ndarray:
        action_count: int = len(self._environment.actions)
        probability_vector: np.ndarray = np.zeros(shape=action_count, dtype=float)

        probability: float = self._environment.one_over_possible_actions[s]

        compatible_actions: np.ndarray = self._environment.s_a_compatibility[s, :]
        probability_vector[compatible_actions] = probability

        return probability_vector

    def _calc_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        probability_matrix = np.zeros(shape=(state_count, action_count), dtype=float)

        probabilities: np.ndarray = self._environment.one_over_possible_actions

        compatible_actions: np.ndarray = self._environment.s_a_compatibility
        probability_matrix[compatible_actions] = probabilities

        return probability_matrix

    # pycharm is asking for this to be implemented even though it's not an abstract method, might be a pycharm bug
    def __setitem__(self, s: int, a: int):
        super().__setitem__(s, a)
