from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction


class Feature(ABC):
    def __init__(self, max_size: Optional[int] = None):
        """
        if sparse implement def _get_x_sparse if not sparse implement _get_x
        :param max_size: maximise size of the feature vector returned (whether sparse or not)
        """
        self._max_size: Optional[int] = max_size
        # is_sparse: whether the return will be just the 1 indices or a full vector, overridden in SparseFeature
        self._is_sparse: bool = False

        # current state or state-action pair
        self._state: Optional[NonTabularState] = None
        self._action: Optional[NonTabularAction] = None

        # current values of item (state or state-action pair)
        self._float_array: np.ndarray = np.array([], dtype=float)
        self._discrete_tuple: tuple = tuple()

    @property
    def max_size(self) -> int:
        if self._max_size is None:
            raise Exception("max_size is not set")
        else:
            return self._max_size

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    def __getitem__(self, item: Union[NonTabularState, tuple[NonTabularState, NonTabularAction]]) -> np.ndarray:
        """returns either the vector as normal or if a sparse feature just the indexes that 1"""
        self.set_values(item)
        return self._get_x()

    def get_x(self, item: Union[NonTabularState, tuple[NonTabularState, NonTabularAction]]) -> np.ndarray:
        """always returns the full feature vector regardless of is_sparse"""
        self.set_values(item)
        return self._get_x()

    def set_values(self, item: Union[NonTabularState, tuple[NonTabularState, NonTabularAction]]):
        if isinstance(item, NonTabularState):
            item: NonTabularState
            self._state = item
            self._float_array, self._discrete_tuple = self._state.values
        else:
            item: tuple[NonTabularState, NonTabularAction]
            self._state, self._action = item
            self._float_array, self._discrete_tuple = self._state.values
            self._discrete_tuple += self._action.values

    @abstractmethod
    def _get_x(self) -> np.ndarray:
        """return the full feature vector"""
        pass
