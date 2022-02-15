from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from mdp.model.environment.state import State


@dataclass(frozen=True)
class NonTabularState(State, ABC):
    __float_array: np.ndarray = field(init=False, hash=False, compare=False)
    __discrete_tuple: tuple = field(init=False, hash=False, compare=False)

    def __post_init__(self):
        # from: https://stackoverflow.com/q/53756788
        object.__setattr__(self, "_NonTabularState__float_array", np.array(self._get_float_list()))
        object.__setattr__(self, "_NonTabularState__discrete_tuple", self._get_discrete_tuple())

    @property
    def values(self) -> tuple[np.ndarray, tuple]:
        return self.__float_array, self.__discrete_tuple

    @abstractmethod
    def _get_float_list(self) -> list[float]:
        pass

    def _get_discrete_tuple(self) -> tuple:
        return tuple()
