from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from mdp import common
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.placeholder_action import PlaceholderAction
from mdp.model.non_tabular.feature.feature import Feature

State = TypeVar('State', bound=NonTabularState)


class StateFunction(Generic[State], ABC):
    def __init__(self,
                 feature: Optional[Feature[State, PlaceholderAction]],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        self._feature: Optional[Feature[State, PlaceholderAction]] = feature
        self._initial_value: float = value_function_parameters.initial_value

    @property
    def has_sparse_feature(self) -> bool:
        """determines whether functions like get_gradient return a vector or a vector of indices"""
        if self._feature:
            return self._feature.is_sparse
        else:
            return False

    @abstractmethod
    def __getitem__(self, state: State) -> float:
        pass
