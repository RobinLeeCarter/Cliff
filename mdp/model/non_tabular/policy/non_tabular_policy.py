from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
    from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment


# specifically non-generic because policies can act in terms of non-specific actions and states
class NonTabularPolicy(ABC):
    def __init__(self, environment: NonTabularEnvironment, policy_parameters: common.PolicyParameters):
        self._environment: NonTabularEnvironment = environment
        self._policy_parameters: common.PolicyParameters = policy_parameters

        # possible actions for a particular state
        self._possible_actions: list[NonTabularAction] = []
        self._all_action_count: int = len(self._environment.actions)
        self._probabilities: np.ndarray = np.zeros(shape=self._all_action_count, dtype=float)

    def __getitem__(self, state: NonTabularState) -> Optional[NonTabularAction]:
        if state.is_terminal:
            return None
        else:
            return self._draw_action(state)

    @abstractmethod
    def _draw_action(self, state: NonTabularState) -> NonTabularAction:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        pass

    @property
    def linked_policy(self) -> NonTabularPolicy:
        """Deterministic partner policy if exists else self"""
        return self

    @abstractmethod
    def get_probability(self, state: NonTabularState, action: NonTabularAction) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, state: NonTabularState) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        pass

