from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import utils
from mdp import common
if TYPE_CHECKING:
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
from mdp.model.policy.tabular.deterministic import Deterministic
from mdp.model.policy.tabular.tabular_policy import TabularPolicy


class EGreedy(TabularPolicy):
    def __init__(self, environment: TabularEnvironment, policy_parameters: common.PolicyParameters):
        super().__init__(environment, policy_parameters)
        self.epsilon: float = self._policy_parameters.epsilon
        greedy_policy_parameters = common.PolicyParameters(
            policy_type=common.PolicyType.DETERMINISTIC,
            store_matrix=False,
        )
        self.greedy_policy: Deterministic = Deterministic(self._environment, greedy_policy_parameters)

    @property
    def linked_policy(self) -> Deterministic:
        return self.greedy_policy

    def set_policy_vector(self, policy_vector: np.ndarray):
        self.greedy_policy.set_policy_vector(policy_vector)
        if self._store_matrix:
            self._policy_matrix = self._calc_policy_matrix()

    def get_policy_vector(self) -> np.ndarray:
        return self.greedy_policy.policy_vector

    def _get_a(self, s: int) -> int:
        if self._store_matrix:
            return utils.p_choice(p=self._policy_matrix[s, :])
        else:
            # could also jit this if needed
            if utils.uniform() > self.epsilon:
                return self.greedy_policy[s]
            else:
                # flat fairly slow at 9ms
                flat = np.flatnonzero(self._environment.s_a_compatibility[s, :])
                i = utils.n_choice(flat.shape[0])
                return flat[i]

    def __setitem__(self, s: int, a: int):
        if self._store_matrix:
            prev_a = self.greedy_policy[s]
            greedy_p = self._policy_matrix[s, prev_a]
            non_greedy_p = self._policy_matrix[s, a]
            self._policy_matrix[s, prev_a] = non_greedy_p
            self._policy_matrix[s, a] = greedy_p
        self.greedy_policy[s] = a
        # print(f"greedy_policy[{s}] = {self.greedy_policy[s]}")

    def _calc_probability(self, s: int, a: int) -> float:
        non_greedy_p = self.epsilon * self._environment.one_over_possible_actions[s]
        if a == self.greedy_policy[s]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p

    def _calc_probability_vector(self, s: int) -> np.ndarray:
        action_count: int = len(self._environment.actions)
        probability_vector: np.ndarray = np.zeros(shape=action_count, dtype=float)

        non_greedy_p: float = self.epsilon * self._environment.one_over_possible_actions[s]
        greedy_p: float = (1 - self.epsilon) + non_greedy_p

        compatible_actions: np.ndarray = self._environment.s_a_compatibility[s, :]
        probability_vector[compatible_actions] = non_greedy_p

        a = self.greedy_policy[s]
        probability_vector[a] = greedy_p

        return probability_vector

    def _calc_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)

        non_greedy_p: np.ndarray = self.epsilon * self._environment.one_over_possible_actions
        greedy_p: np.ndarray = (1 - self.epsilon) + non_greedy_p
        # greedy_p to zero when non_greedy_p is zero (when no actions are allowed)
        greedy_p[non_greedy_p == 0.0] = 0.0

        # broadcast (|S|,) to (|S|,|A|)
        non_greedy_p_broadcast = np.broadcast_to(non_greedy_p[:, np.newaxis], shape=policy_matrix.shape)
        compatible_actions: np.ndarray = self._environment.s_a_compatibility
        policy_matrix[compatible_actions] = non_greedy_p_broadcast[compatible_actions]

        i = np.arange(state_count)
        policy_vector = self.greedy_policy.policy_vector
        policy_matrix[i, policy_vector] = greedy_p

        return policy_matrix
