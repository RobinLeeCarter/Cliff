from __future__ import annotations
from typing import TYPE_CHECKING, Type, TypeVar, Generic

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.feature import Feature
from mdp import common
from mdp.model.non_tabular.value_function.state_function import StateFunction
from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction
from mdp.model.non_tabular.value_function.linear_state_function import LinearStateFunction
from mdp.model.non_tabular.value_function.linear_state_action_function import LinearStateActionFunction

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class ValueFunctionFactory(Generic[State, Action]):
    def __init__(self):
        v = common.ValueFunctionType
        self._state_function_lookup: dict[v, Type[StateFunction]] = {
            v.LINEAR_STATE: LinearStateFunction,
        }
        self._state_action_function_lookup: dict[v, Type[StateActionFunction]] = {
            v.LINEAR_STATE_ACTION: LinearStateActionFunction
        }

    def create_state_function(self,
                              feature: Feature,
                              value_function_parameters: common.ValueFunctionParameters) \
            -> StateFunction:
        value_function_type: common.ValueFunctionType = value_function_parameters.value_function_type
        type_of_value_function: Type[StateFunction] = self._state_function_lookup[value_function_type]
        value_function: StateFunction = type_of_value_function[State](feature, value_function_parameters)
        return value_function

    def create_state_action_function(self,
                                     feature: Feature,
                                     value_function_parameters: common.ValueFunctionParameters) \
            -> StateActionFunction:
        value_function_type: common.ValueFunctionType = value_function_parameters.value_function_type
        type_of_value_function: Type[StateActionFunction] = self._state_action_function_lookup[value_function_type]
        value_function: StateActionFunction = type_of_value_function[State, Action](feature, value_function_parameters)
        return value_function
