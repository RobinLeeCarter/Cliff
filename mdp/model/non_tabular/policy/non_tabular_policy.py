from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, TypeVar, Generic

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.feature.feature import Feature
    from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction

from mdp.model.base.policy.base_policy import BasePolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


# specifically non-generic because policies can act in terms of non-specific actions and states
class NonTabularPolicy(Generic[State, Action], BasePolicy, ABC):
    policy_type: common.PolicyType

    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters):
        super().__init__(environment, policy_parameters)
        self._environment: NonTabularEnvironment[State, Action] = environment

        # possible actions for a particular state
        self._possible_actions: list[Action] = []
        self._all_action_count: int = len(self._environment.actions)
        self._probabilities: np.ndarray = np.zeros(shape=self._all_action_count, dtype=float)

        self.requires_feature: bool = False
        self.requires_q: bool = False

    def set_feature(self, feature: Feature[State, Action]):
        raise Exception("NotImplemented")

    def set_state_action_function(self, state_action_function: StateActionFunction[State, Action]):
        raise Exception("NotImplemented")

    def __getitem__(self, state: State) -> Optional[Action]:
        if state.is_terminal:
            return None
        else:
            return self._draw_action(state)

    @abstractmethod
    def _draw_action(self, state: State) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        pass

    @property
    def linked_policy(self) -> NonTabularPolicy[State, Action]:
        """Deterministic partner policy if exists else self"""
        return self

    @abstractmethod
    def get_probability(self, state: State, action: Action) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, state: State) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        pass

