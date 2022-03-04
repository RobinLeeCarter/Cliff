from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cache

import numpy as np

from mdp.model.general.environment.general_action import GeneralAction


@dataclass(frozen=True)
class NonTabularAction(GeneralAction, ABC):
    # Restore as property once Pycharm issue is fixed?: https://youtrack.jetbrains.com/issue/PY-48338
    # @property
    @cache
    def categories(self) -> np.ndarray:
        return np.array(self._get_categories(), dtype=object)

    @abstractmethod
    def _get_categories(self) -> list:
        ...
