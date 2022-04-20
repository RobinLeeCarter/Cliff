from __future__ import annotations
from typing import TYPE_CHECKING, Type, TypeVar, Generic, Optional


if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp import common
from mdp.model.non_tabular.value_function.base_value_function import BaseValueFunction
from mdp.model.non_tabular.value_function.state.state_function import StateFunction
from mdp.model.non_tabular.value_function.state_action.state_action_function import StateActionFunction
from mdp.model.non_tabular.value_function.state.linear_state_function import LinearStateFunction
from mdp.model.non_tabular.value_function.state_action.linear_state_action_function import LinearStateActionFunction
from mdp.model.non_tabular.value_function.state_action.linear_state_action_shared_weights import \
    LinearStateActionSharedWeights

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class ValueFunctionFactory(Generic[State, Action]):
    """Non-tabular case only"""
    def create_state_function(self,
                              feature: Optional[BaseFeature],
                              value_function_parameters: common.ValueFunctionParameters) \
            -> StateFunction:
        value_function_type: common.ValueFunctionType = value_function_parameters.value_function_type
        type_of_value_function: Type[BaseValueFunction] = BaseValueFunction.type_registry[value_function_type]
        assert issubclass(type_of_value_function, StateFunction)
        value_function: StateFunction = type_of_value_function[State](feature, value_function_parameters)
        return value_function

    def create_state_action_function(self,
                                     feature: Optional[BaseFeature],
                                     value_function_parameters: common.ValueFunctionParameters) \
            -> StateActionFunction:
        value_function_type: common.ValueFunctionType = value_function_parameters.value_function_type
        type_of_value_function: Type[BaseValueFunction] = BaseValueFunction.type_registry[value_function_type]
        assert issubclass(type_of_value_function, StateActionFunction)
        value_function: StateActionFunction = type_of_value_function[State, Action](feature, value_function_parameters)
        return value_function


def __dummy():
    """Stops Pycharm objecting to imports. The imports are needed to generate the registry."""
    return [
        LinearStateFunction,
        LinearStateActionFunction,
        LinearStateActionSharedWeights,
    ]
