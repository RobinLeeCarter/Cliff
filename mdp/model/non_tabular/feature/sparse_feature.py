from __future__ import annotations
from typing import Union, TypeVar, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np


if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dims import Dims
from mdp import common
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.feature.feature import Feature

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class SparseFeature(Feature[State, Action], ABC):
    """
    Any feature that by default should return a sparse vector of just the indices that are 1
    State Aggregation is a sparse feature
    Coarse-coding is also a sparse feature (if the features are binary)
    """
    def __init__(self, dims: Dims, feature_parameters: common.FeatureParameters):
        super().__init__(dims, feature_parameters)
        self._is_sparse: bool = True

    def __getitem__(self, item: Union[State, tuple[State, Action]]) -> np.ndarray:
        """returns either the vector as normal or if a sparse feature just the indexes that 1"""
        self.unpack_item(item)
        return self._get_sparse_vector()

    def _get_full_vector(self) -> np.ndarray:
        """return the full x vector"""
        if self._max_size:
            sparse_vector = self.get_vector
            full_vector = np.zeros(shape=self._max_size, dtype=np.int)
            full_vector[sparse_vector] = 1
            return full_vector
        else:
            raise Exception("Size of x not specified")

    def get_vector(self) -> np.ndarray:
        if not self._vector:
            # can't use cached version so calculate
            self._vector = self._get_sparse_vector()
        return self._vector

    def dot_product_full_vector(self, full_vector: np.ndarray) -> float:
        return float(np.sum(full_vector[self.get_vector()]))

    @abstractmethod
    def _get_sparse_vector(self) -> np.ndarray:
        """return just the indexes of x which are 1 (rest are 0) using unpacked values"""
        pass
