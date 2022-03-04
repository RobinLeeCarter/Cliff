from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState


class StateFunction(ABC):
    """at it's most general a State Function returns a scalar from a state, how it does that is up to it"""
    @abstractmethod
    def __getitem__(self, state: NonTabularState) -> float:
        pass
