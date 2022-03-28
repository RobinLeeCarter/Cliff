from __future__ import annotations
from typing import Optional, TypeVar, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dims import Dims
from mdp.model.non_tabular.feature.sparse_feature import SparseFeature
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class CompoundSparseFeature(SparseFeature[State, Action]):
    """untested"""
    def __init__(self, dims: Dims, features: list[SparseFeature[State, Action]]):
        """
        A list of features (with max_size already set) itself forming a feature with results concatenated
        :param features: list of
        """
        super().__init__(dims)
        self._features: list[SparseFeature[State, Action]] = features
        self._max_size: Optional[int] = sum(feature.max_size for feature in self._features)

    def _get_sparse_vector(self) -> np.ndarray:
        """return the full feature vector using unpacked values"""
        for feature in self._features:
            feature.set_state(self._state)
            if self._action:
                feature.set_action(self._action)
        results: list[np.ndarray] = [feature.get_vector() for feature in self._features]
        concat_results: np.ndarray = np.concatenate(results)
        return concat_results
