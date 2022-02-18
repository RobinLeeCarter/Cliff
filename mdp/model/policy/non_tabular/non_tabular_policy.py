from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.action import Action
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment


class NonTabularPolicy(abc.ABC):
    def __init__(self, environment: NonTabularEnvironment, policy_parameters: common.PolicyParameters):
        self._environment: NonTabularEnvironment = environment
        self._policy_parameters: common.PolicyParameters = policy_parameters

    def __getitem__(self, state: NonTabularState) -> Optional[Action]:
        if state.is_terminal:
            return None
        else:
            return self._draw_action(state)

    @abc.abstractmethod
    def _draw_action(self, state: NonTabularState) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        pass

    @property
    def linked_policy(self) -> NonTabularPolicy:
        """Deterministic partner policy if exists else self"""
        return self

    @abc.abstractmethod
    def get_probability(self, state: NonTabularState, action: NonTabularAction) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        pass

    @abc.abstractmethod
    def get_action_probabilities(self, state: NonTabularState) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        pass
