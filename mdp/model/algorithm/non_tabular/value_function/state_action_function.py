from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction


class StateActionFunction(ABC):
    """at it's most general a State Function returns a scalar from a state, how it does that is up to it"""
    @abstractmethod
    def __getitem__(self, state: NonTabularState, action: NonTabularAction) -> float:
        pass

    @abstractmethod
    def get_action_values(self, state: NonTabularState, actions: list[NonTabularAction]) -> np.ndarray:
        """efficiently calculate multiple state-action values using the fact that the state is the same for all"""
        pass
