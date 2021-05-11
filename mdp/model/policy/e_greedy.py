from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from mdp import common
if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.policy import policy
from mdp.model.policy import random, deterministic


class EGreedy(random.Random):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self.greedy_policy: deterministic.Deterministic =\
            deterministic.Deterministic(self._environment, self._policy_parameters)
        self.epsilon = self._policy_parameters.epsilon

    def _get_action(self, s: int) -> int:
        if common.rng.uniform() > self.epsilon:
            return self.greedy_policy[s]
        else:
            return random.Random._get_action(self, s)

    def __setitem__(self, s: int, a: int):
        self.greedy_policy[s] = a

    @property
    def linked_policy(self) -> policy.Policy:
        return self.greedy_policy

    def get_probability(self, s: int, a: int) -> float:
        non_greedy_p = self.epsilon * self._environment.one_over_possible_actions[s]
        if a == self.greedy_policy[s]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p

    def get_probability_vector(self, s: int) -> np.ndarray:
        # TODO: Decide whether to maintain probability_matrix as policy updates
        action_count: int = len(self._environment.actions)
        probability_vector: np.ndarray = np.zeros(shape=action_count, dtype=float)

        non_greedy_p: float = self.epsilon * self._environment.one_over_possible_actions[s]
        greedy_p: float = (1 - self.epsilon) + non_greedy_p

        compatible_actions: np.ndarray = self._environment.s_a_compatibility[s, :]
        probability_vector[compatible_actions] = non_greedy_p

        a = self.greedy_policy[s]
        probability_vector[a] = greedy_p

        return probability_vector

    def get_probability_matrix(self) -> np.ndarray:
        # TODO: Decide whether to maintain probability_matrix as policy updates
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        probability_matrix = np.zeros(shape=(state_count, action_count), dtype=float)

        non_greedy_p: np.ndarray = self.epsilon * self._environment.one_over_possible_actions
        greedy_p: np.ndarray = (1 - self.epsilon) + non_greedy_p

        compatible_actions: np.ndarray = self._environment.s_a_compatibility
        probability_matrix[compatible_actions] = non_greedy_p

        i = np.arange(state_count)
        policy_vector = self.greedy_policy.policy_vector
        probability_matrix[i, policy_vector] = greedy_p

        return probability_matrix

    def get_policy_vector(self) -> np.ndarray:
        return self.greedy_policy.policy_vector

    def set_policy_vector(self, policy_vector: np.ndarray):
        self.greedy_policy.policy_vector = policy_vector
