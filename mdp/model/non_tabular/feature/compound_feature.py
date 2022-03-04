from __future__ import annotations
from typing import Optional, TypeVar

import numpy as np

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.feature.feature import Feature

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class CompoundFeature(Feature[State, Action]):
    """untested"""
    def __init__(self, features: list[Feature[State, Action]]):
        """
        A list of features (with max_size already set) itself forming a feature with results concatenated
        :param features: list of
        """
        super().__init__()
        self._features = features
        self._max_size: Optional[int] = sum(feature.max_size for feature in self._features)

    def _get_full_vector(self) -> np.ndarray:
        """return the full feature vector using unpacked values"""
        for feature in self._features:
            feature.set_state(self._state)
            if self._action:
                feature.set_action(self._action)
        results: list[np.ndarray] = [feature.get_vector() for feature in self._features]
        concat_results: np.ndarray = np.concatenate(results)
        return concat_results
