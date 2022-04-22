from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

import numpy as np

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


@dataclass
class RewardStateAction(Generic[State, Action]):
    r: float
    state: State
    action: Optional[Action]
    feature_vector: Optional[np.ndarray] = None

    @property
    def tuple(self) -> tuple[float, State, Optional[Action]]:
        return self.r, self.state, self.action


Trajectory = list[RewardStateAction]
