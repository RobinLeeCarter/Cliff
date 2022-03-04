from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
    from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction


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

        # cached value of feature vector, set to None if not up-to-date
        self._vector: Optional[np.ndarray] = None

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
        self.unpack_item(item)
        return self._get_full_vector()

    def get_full_x(self, item: Union[NonTabularState, tuple[NonTabularState, NonTabularAction]]) -> np.ndarray:
        """always returns the full feature vector regardless of is_sparse"""
        self.unpack_item(item)
        return self._get_full_vector()

    def unpack_item(self, item: Union[NonTabularState, tuple[NonTabularState, NonTabularAction]]):
        if isinstance(item, tuple):
            item: tuple[NonTabularState, NonTabularAction]
            self.state, self.action = item
        else:
            item: NonTabularState
            self.state = item

    @property
    def state(self) -> NonTabularState:
        return self._state

    @state.setter
    def state(self, state: NonTabularState):
        self._state = state
        self._vector = None
        self._do_state_computation()

    def _do_state_computation(self):
        pass

    @property
    def action(self) -> NonTabularAction:
        return self._action

    @action.setter
    def action(self, action: NonTabularAction):
        self._action = action
        self._vector = None
        self._do_action_computation()

    def _do_action_computation(self):
        pass

    @property
    def vector(self) -> np.ndarray:
        if not self._vector:
            self._vector = self._get_full_vector()
        return self._vector

    def dot_product_full_vector(self, full_vector: np.ndarray) -> float:
        return float(np.dot(full_vector, self.vector))

    @abstractmethod
    def _get_full_vector(self) -> np.ndarray:
        """return the full feature vector using state and action values"""
        pass
