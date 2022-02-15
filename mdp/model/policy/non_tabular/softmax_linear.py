from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.action import Action
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
    from mdp.model.feature_construction.feature import Feature
from mdp.model.policy.non_tabular.non_tabular_policy import NonTabularPolicy


class SoftmaxLinear(NonTabularPolicy):
    def __init__(self,
                 environment: NonTabularEnvironment,
                 policy_parameters: common.PolicyParameters,
                 feature: Feature,
                 initial_value_theta: float = 0.0
                 ):
        super().__init__(environment, policy_parameters)
        self._feature: Feature = feature
        self._theta = np.full(shape=feature.max_size, fill_value=initial_value_theta, dtype=float)

    def _draw_action(self, state: NonTabularState) -> Action:
        for action in self._environment.actions:
            preference = self._feature.dot_product(self._theta, state, action)
            x: np.ndarray = self._feature[state, action]
            if self._feature.is_sparse:
                dot_product: np.ndarray = self._theta[x]
            else:
                dot_product: np.ndarray = np.dot(self._theta, x)
            pass

    # @property
    # def linked_policy(self) -> NonTabularPolicy:
    #     """Deterministic partner policy if exists else self"""
    #     return self
    #
    # def get_probability(self, s: int, a: int) -> float:
    #     if self._store_matrix:
    #         return self._policy_matrix[s, a]
    #     else:
    #         return self._calc_probability(s, a)
    #
    # def get_probability_vector(self, s: int) -> np.ndarray:
    #     if self._store_matrix:
    #         return self._policy_matrix[s, :]
    #     else:
    #         return self._calc_probability_vector(s)
    #
    # def get_probability_matrix(self) -> np.ndarray:
    #     if self._store_matrix:
    #         return self._policy_matrix
    #     else:
    #         return self._calc_policy_matrix()
    #
    # @abc.abstractmethod
    # def _calc_probability(self, s: int, a: int) -> float:
    #     pass
    #
    # def _calc_probability_vector(self, s: int) -> np.ndarray:
    #     action_count = len(self._environment.actions)
    #     probability_vector = np.zeros(shape=action_count, dtype=float)
    #     for a in range(action_count):
    #         if self._environment.s_a_compatibility[s, a]:
    #             probability_vector[s, a] = self._calc_probability(s, a)
    #     return probability_vector
    #
    # def _calc_policy_matrix(self) -> np.ndarray:
    #     state_count = len(self._environment.states)
    #     action_count = len(self._environment.actions)
    #     policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
    #     for s in range(state_count):
    #         policy_matrix[s, :] = self.get_probability_vector(s)
    #         # for a in range(action_count):
    #         #     if self._environment.s_a_compatibility[s, a]:
    #         #         policy_matrix[s, a] = self.get_probability(s, a)
    #     return policy_matrix
    #
    # def get_policy_vector(self) -> np.ndarray:
    #     pass
    #
    # def set_policy_vector(self, policy_vector: np.ndarray):
    #     pass
