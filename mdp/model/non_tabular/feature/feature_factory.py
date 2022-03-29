from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Type, TypeVar, Generic, Optional

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.feature import Feature
    from mdp.model.non_tabular.environment.dimension.dims import Dims
from mdp import common
from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class FeatureFactory(Generic[State, Action]):
    def __init__(self, dims: Dims):
        self._dims: Dims = dims

        f = common.FeatureType
        self._feature_lookup: dict[f, Type[Feature]] = {
            f.TILE_CODING: TileCoding
        }

    def create(self, feature_parameters: Optional[common.FeatureParameters]) -> Feature:
        feature_type: common.FeatureType = feature_parameters.feature_type
        type_of_feature: Type[Feature] = self._feature_lookup[feature_type]
        feature: Feature = type_of_feature[State, Action](self._dims, feature_parameters)
        return feature
