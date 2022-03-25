from __future__ import annotations
from typing import Generic, TypeVar
from abc import ABC, abstractmethod

import numpy as np

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class StateActionFunction(Generic[State, Action], ABC):
    """at it's most general a State Function returns a scalar from a state, how it does that is up to it"""
    @abstractmethod
    def __getitem__(self, state: State, action: Action) -> float:
        pass

    @abstractmethod
    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        """efficiently calculate multiple state-action values using the fact that the state is the same for all"""
        pass
