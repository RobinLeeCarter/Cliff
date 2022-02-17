from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
    from mdp.model.feature.compound_feature import CompoundFeature
from mdp.model.feature.feature import Feature


class SparseFeature(Feature, ABC):
    """
    Any feature that by default should return a sparse vector of just the indices that are 1
    State Aggregation is a sparse feature
    Coarse-coding is also a sparse feature (if the features are binary)
    """
    def __init__(self, max_size: Optional[int] = None):
        super().__init__(max_size)
        self._is_sparse: bool = True

    def __getitem__(self, item: Union[NonTabularState, tuple[NonTabularState, NonTabularAction]]) -> np.ndarray:
        """returns either the vector as normal or if a sparse feature just the indexes that 1"""
        self.unpack_values(item)
        return self._get_x_sparse()

    def _get_x(self) -> np.ndarray:
        """return the full x vector"""
        if self._max_size:
            x_sparse: np.ndarray = self._get_x_sparse()
            x = np.zeros(shape=self._max_size, dtype=np.int)
            x[x_sparse] = 1
            return x
        else:
            raise Exception("Size of x not specified")

    # def copy_and_get_x_sparse(self, compound_feature: CompoundFeature) -> np.ndarray:
    #     """copy the unpacked values from the compound feature and then _get_x"""
    #     self._state, self._action, self._state_floats, self._state_categories, self._action_categories = \
    #         compound_feature.unpacked_values
    #     return self._get_x_sparse()

    @abstractmethod
    def _get_x_sparse(self) -> np.ndarray:
        """return just the indexes of x which are 1 (rest are 0) using unpacked values"""
        pass
