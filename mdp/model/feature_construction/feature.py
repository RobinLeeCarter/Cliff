from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.state import State


class Feature(ABC):
    @abstractmethod
    def get_x(self, state: State) -> np.ndarray:
        pass
