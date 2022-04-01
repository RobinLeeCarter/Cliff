from __future__ import annotations
from abc import ABC
from typing import Optional, Type

from mdp import common
from mdp.model.non_tabular.feature.base_feature import BaseFeature


class BaseValueFunction(ABC):
    type_registry: dict[common.ValueFunctionType, Type[BaseValueFunction]] = {}

    def __init_subclass__(cls,
                          value_function_type: Optional[common.ValueFunctionType] = None,
                          **kwargs):
        super().__init_subclass__(**kwargs)
        if value_function_type:
            BaseValueFunction.type_registry[value_function_type] = cls

    def __init__(self,
                 feature: Optional[BaseFeature],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        self._feature: Optional[BaseFeature] = feature
        self._initial_value: float = value_function_parameters.initial_value

    @property
    def has_sparse_feature(self) -> bool:
        """determines whether functions like get_gradient return a vector or a vector of indices"""
        if self._feature:
            return self._feature.is_sparse
        else:
            return False
