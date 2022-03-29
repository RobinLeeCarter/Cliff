from __future__ import annotations
from typing import Generic, TypeVar
from abc import ABC, abstractmethod

import numpy as np

from mdp import common
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.feature.feature import Feature

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class StateActionFunction(Generic[State, Action], ABC):
    def __init__(self,
                 feature: Feature[State, Action],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        self.feature: Feature[State, Action] = feature
        self.initial_value: float = value_function_parameters.initial_value

    @property
    def has_sparse_feature(self) -> bool:
        """determines whether functions like get_gradient return a vector or a vector of indices"""
        return self.feature.is_sparse

    @abstractmethod
    def __getitem__(self, state: State, action: Action) -> float:
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
