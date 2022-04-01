from __future__ import annotations
from typing import TYPE_CHECKING, Type, TypeVar, Generic, Optional

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dims import Dims
from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp import common

from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction


State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class FeatureFactory(Generic[State, Action]):
    def __init__(self, dims: Dims):
        self._dims: Dims = dims

    def create(self, feature_parameters: Optional[common.FeatureParameters]) -> BaseFeature:
        feature_type: common.FeatureType = feature_parameters.feature_type
        type_of_feature: Type[BaseFeature] = BaseFeature.type_registry[feature_type]
        feature: BaseFeature = type_of_feature[State, Action](self._dims, feature_parameters)
        return feature


def __dummy():
    """Stops Pycharm objecting to imports. The imports are needed to generate the registry."""
    return [
        TileCoding
    ]
