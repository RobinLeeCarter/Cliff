from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState

State = TypeVar('State', bound=NonTabularState)


class StateFunction(Generic[State], ABC):
    """at it's most general a State Function returns a scalar from a state, how it does that is up to it"""
    @abstractmethod
    def __getitem__(self, state: State) -> float:
        pass
