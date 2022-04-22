from __future__ import annotations
from abc import ABC
from typing import Optional, Type

from mdp import common
from mdp.model.non_tabular.feature.base_feature import BaseFeature


class BaseValueFunction(ABC):
    type_registry: dict[common.ValueFunctionType, Type[BaseValueFunction]] = {}
    has_feature_matrix: bool = False
    shared_weights: bool = False

    def __init_subclass__(cls,
                          value_function_type: Optional[common.ValueFunctionType] = None,
                          has_feature_matrix: bool = False,
                          shared_weights: bool = False,
                          **kwargs):
        super().__init_subclass__(**kwargs)
        if value_function_type:
            BaseValueFunction.type_registry[value_function_type] = cls
        if has_feature_matrix:
            cls.has_feature_matrix = has_feature_matrix
        if shared_weights:
            cls.shared_weights = shared_weights

    def __init__(self,
                 feature: Optional[BaseFeature],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        self._feature: Optional[BaseFeature] = feature
        self._has_sparse_feature: bool = False
        if self._feature:
            self._has_sparse_feature = self._feature.is_sparse
        self._initial_value: float = value_function_parameters.initial_value

    @property
    def has_sparse_feature(self) -> bool:
        """determines whether functions like get_gradient return a vector or a vector of indices"""
        return self._has_sparse_feature
