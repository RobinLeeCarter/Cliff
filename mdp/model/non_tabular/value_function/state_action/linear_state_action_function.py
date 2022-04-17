from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional

import numpy as np

from mdp import common

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.value_function.state_action.state_action_function import StateActionFunction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class LinearStateActionFunction(StateActionFunction[State, Action],
                                value_function_type=common.ValueFunctionType.LINEAR_STATE_ACTION):
    def __init__(self,
                 feature: BaseFeature[State, Action],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)

        self.size: int = self._feature.max_size
        # weights
        self.w: np.ndarray = np.full(shape=self.size, fill_value=self._initial_value, dtype=float)
        self._feature_vector: Optional[np.ndarray] = None

        self.delta_w: Optional[np.ndarray] = None
        self.original_w: Optional[np.ndarray] = None
        if value_function_parameters.requires_delta_w:
            self.delta_w: np.ndarray = np.zeros_like(self.w)
            self.original_w: np.ndarray = self.w.copy()

    def __getitem__(self, state_action: tuple[State, Action]) -> float:
        state, action = state_action
        value: float
        if state.is_terminal:
            value = 0.0
        else:
            self._feature_vector = self._feature[state, action]
            value = self.calc_value(self._feature_vector)
        return value

    def get_gradient(self, state: State, action: Action) -> np.ndarray:
        self._feature_vector = self._feature[state, action]
        gradient = self.calc_gradient(self._feature_vector)
        return gradient

    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        feature_matrix: np.ndarray = self._feature.get_matrix(state, actions)
        values_array: np.ndarray = self._feature.matrix_product(feature_matrix, self.w)
        return values_array

    # def get_action_values3(self, state: State, actions: list[Action]) -> np.ndarray:
    #     values_array: np.ndarray = self._feature.get_dot_products(state, actions, self.w)
    #     return values_array

    def calc_value(self, feature_vector: np.ndarray) -> float:
        value: float = self._feature.dot_product(feature_vector, self.w)
        return value

    def calc_gradient(self, feature_vector: np.ndarray) -> np.ndarray:
        return feature_vector

    # Weights
    def update_weights(self, delta_w: np.ndarray):
        self.w += delta_w

    def update_weights_sparse(self, indices: np.ndarray, delta_w: float):
        self.w[indices] += delta_w

    def copy_w(self):
        self.original_w: np.ndarray = self.w.copy()

    # Delta weights
    def get_delta_weights(self) -> np.ndarray:
        # return self.delta_w
        return self.w - self.original_w

    def apply_delta_weights(self):
        self.w += self.delta_w
