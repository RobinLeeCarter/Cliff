from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.environment import Environment


class Policy(abc.ABC):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        self._environment = environment_
        self._policy_parameters: common.PolicyParameters = policy_parameters

    def __getitem__(self, s: int) -> Optional[int]:
        if self._environment.is_terminal[s]:
            return None
        else:
            # this of course will go to the level in inheritance hierarchy set by self
            return self._get_action(s)

    def __setitem__(self, s: int, a: int):
        raise NotImplementedError(f"__setitem__ not implemented for Policy: {type(self)}")

    @property
    def linked_policy(self) -> Policy:
        """Deterministic partner policy if exists else self"""
        return self

    @abc.abstractmethod
    def _get_action(self, s: int) -> Optional[int]:
        pass

    @abc.abstractmethod
    def get_probability(self, s: int, a: int) -> float:
        pass

    def get_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
        for s in range(state_count):
            for a in range(action_count):
                if self._environment.s_a_compatibility[s, a]:
                    policy_matrix[s, a] = self.get_probability(s, a)
        return policy_matrix

    def set_policy_vector(self, policy_matrix: np.ndarray, update_dict: bool = True):
        pass
