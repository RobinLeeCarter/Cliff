from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cache

import numpy as np

from mdp.model.general.environment.general_state import GeneralState


@dataclass(frozen=True)
class NonTabularState(GeneralState, ABC):
    # Restore as property once Pycharm issue is fixed?: https://youtrack.jetbrains.com/issue/PY-48338
    # @property
    @cache
    def floats(self) -> np.ndarray:
        return np.array(self._get_floats(), dtype=float)

    # Restore as property once Pycharm issue is fixed?: https://youtrack.jetbrains.com/issue/PY-48338
    # @property
    @cache
    def categories(self) -> np.ndarray:
        return np.array(self._get_categories(), dtype=object)

    @abstractmethod
    def _get_floats(self) -> list[float]:
        ...

    def _get_categories(self) -> list:
        return []


