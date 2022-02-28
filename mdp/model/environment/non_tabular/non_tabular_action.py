from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cache

import numpy as np

from mdp.model.environment.action import Action


@dataclass(frozen=True)
class NonTabularAction(Action, ABC):
    @property
    @cache
    def categories(self) -> np.ndarray:
        return np.array(self._get_categories(), dtype=object)

    @abstractmethod
    def _get_categories(self) -> list:
        ...
