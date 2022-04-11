from __future__ import annotations
from typing import Generic, TypeVar, Optional
from abc import ABC, abstractmethod

import numpy as np

from mdp import common
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.value_function.base_value_function import BaseValueFunction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class StateActionFunction(Generic[State, Action], BaseValueFunction, ABC):
    def __init__(self,
                 feature: Optional[BaseFeature[State, Action]],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)
        self._feature: Optional[BaseFeature[State, Action]] = feature

    @abstractmethod
    def __getitem__(self, state_action: tuple[State, Action]) -> float:
        state, action = state_action
        if state.is_terminal:
            return 0.0
        else:
            raise NotImplementedError

    @abstractmethod
    def get_gradient(self, state: State, action: Action) -> np.ndarray:
        pass

    @abstractmethod
    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        """efficiently calculate multiple state-action values using the fact that the state is the same for all"""
        pass

    @abstractmethod
    def update_weights(self, delta_w: np.ndarray):
        pass

    @abstractmethod
    def update_weights_sparse(self, indices: np.ndarray, delta_w: float):
        pass

    def copy_w(self):
        raise Exception("Not implemented")

    # Delta weights
    def reset_delta_w(self):
        raise Exception("Not implemented")

    def update_delta_weights(self, delta_w: np.ndarray):
        raise Exception("Not implemented")

    def update_delta_weights_sparse(self, indices: np.ndarray, delta_w: float):
        raise Exception("Not implemented")

    def get_delta_weights(self) -> np.ndarray:
        raise Exception("Not implemented")

    def apply_delta_weights(self):
        raise Exception("Not implemented")
