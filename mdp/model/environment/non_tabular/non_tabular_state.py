from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cache

import numpy as np

from mdp.model.environment.state import State


@dataclass(frozen=True)
class NonTabularState(State, ABC):
    @property
    @cache
    def floats(self) -> np.ndarray:
        return np.array(self._get_floats(), dtype=float)

    @property
    @cache
    def categories(self) -> np.ndarray:
        return np.array(self._get_categories(), dtype=object)

    @abstractmethod
    def _get_floats(self) -> list[float]:
        ...

    def _get_categories(self) -> list:
        return []


