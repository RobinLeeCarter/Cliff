from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from mdp import common
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.placeholder_action import PlaceholderAction
from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.value_function.base_value_function import BaseValueFunction

State = TypeVar('State', bound=NonTabularState)


class StateFunction(Generic[State], BaseValueFunction, ABC):
    def __init__(self,
                 feature: Optional[BaseFeature[State, PlaceholderAction]],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)
        self._feature: Optional[BaseFeature[State, PlaceholderAction]] = feature

    @abstractmethod
    def __getitem__(self, state: State) -> float:
        pass
