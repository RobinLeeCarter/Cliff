from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.tabular.tabular_action import TabularAction
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
from mdp.model.policy.general_policy import GeneralPolicy


class TabularPolicy(GeneralPolicy, ABC):
    def __init__(self, environment: TabularEnvironment, policy_parameters: common.PolicyParameters):
        super().__init__(environment, policy_parameters)
        self._environment: TabularEnvironment = environment
        self._policy_parameters: common.PolicyParameters = policy_parameters

        self._store_matrix: bool = self._policy_parameters.store_matrix
        self._policy_matrix: Optional[np.ndarray] = None

    def zero_state_action(self):
        if self._store_matrix:
            state_count = len(self._environment.states)
            action_count = len(self._environment.actions)
            self._policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)

    def __getitem__(self, s: int) -> int:
        if self._environment.is_terminal[s]:
            return 0        # None
        else:
            return self._get_a(s)

    def __setitem__(self, s: int, a: int):
        raise NotImplementedError(f"__setitem__ not implemented for Policy: {type(self)}")

    def get_action(self, s: int) -> TabularAction:
        return self._environment.actions[self._get_a(s)]

    def set_action(self, s: int, action: TabularAction):
        a = self._environment.action_index[action]
        # print(s, action, a)
        self.__setitem__(s, a)

    @property
    def linked_policy(self) -> TabularPolicy:
        """Deterministic partner policy if exists else self"""
        return self

    @abstractmethod
    def _get_a(self, s: int) -> int:
        pass

    def refresh_policy_matrix(self):
        if self._store_matrix:
            self._policy_matrix = self._calc_policy_matrix()

    def get_probability(self, s: int, a: int) -> float:
        if self._store_matrix:
            return self._policy_matrix[s, a]
        else:
            return self._calc_probability(s, a)

    def get_probability_vector(self, s: int) -> np.ndarray:
        if self._store_matrix:
            return self._policy_matrix[s, :]
        else:
            return self._calc_probability_vector(s)

    def get_probability_matrix(self) -> np.ndarray:
        if self._store_matrix:
            return self._policy_matrix
        else:
            return self._calc_policy_matrix()

    @abstractmethod
    def _calc_probability(self, s: int, a: int) -> float:
        pass

    def _calc_probability_vector(self, s: int) -> np.ndarray:
        action_count = len(self._environment.actions)
        probability_vector = np.zeros(shape=action_count, dtype=float)
        for a in range(action_count):
            if self._environment.s_a_compatibility[s, a]:
                probability_vector[s, a] = self._calc_probability(s, a)
        return probability_vector

    def _calc_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
        for s in range(state_count):
            policy_matrix[s, :] = self.get_probability_vector(s)
            # for a in range(action_count):
            #     if self._environment.s_a_compatibility[s, a]:
            #         policy_matrix[s, a] = self.get_probability(s, a)
        return policy_matrix

    def get_policy_vector(self) -> np.ndarray:
        pass

    def set_policy_vector(self, policy_vector: np.ndarray):
        pass
