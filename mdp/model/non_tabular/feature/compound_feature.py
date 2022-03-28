from __future__ import annotations
from typing import Optional, TypeVar, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dims import Dims
    from mdp.model.non_tabular.feature.compound_feature_parameters import CompoundFeatureParameters
from mdp.model.non_tabular.feature.feature_factory import FeatureFactory
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.feature.feature import Feature

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class CompoundFeature(Feature[State, Action]):
    """untested"""
    def __init__(self, dims: Dims, compound_feature_parameters: CompoundFeatureParameters):
        """
        A list of features (with max_size already set) itself forming a feature with results concatenated
        """
        super().__init__(dims, compound_feature_parameters)
        feature_factory: FeatureFactory = FeatureFactory(dims)
        self._features: list[Feature[State, Action]] = \
            [feature_factory.create(feature_parameters)
             for feature_parameters in compound_feature_parameters.feature_parameters_list]
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
