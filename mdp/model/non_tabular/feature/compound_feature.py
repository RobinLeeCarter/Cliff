from __future__ import annotations
from typing import Optional

import numpy as np

from mdp.model.non_tabular.feature.feature import Feature


class CompoundFeature(Feature):
    """untested"""
    def __init__(self, features: list[Feature]):
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
            feature.state = self._state
            if self._action:
                feature.action = self._action
        results: list[np.ndarray] = [feature.vector for feature in self._features]
        concat_results: np.ndarray = np.concatenate(results)
        return concat_results
