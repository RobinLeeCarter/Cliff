from __future__ import annotations

from multiprocessing import Lock
from typing import TYPE_CHECKING, TypeVar, Optional

import numpy as np

from utils import SharedArrayWrapper
from mdp import common


if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.value_function.state_action.linear_state_action_function import LinearStateActionFunction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class LinearStateActionSharedWeights(LinearStateActionFunction[State, Action],
                                     value_function_type=common.ValueFunctionType.LINEAR_STATE_ACTION_SHARED_WEIGHTS):
    def __init__(self,
                 feature: BaseFeature[State, Action],
                 value_function_parameters: common.ValueFunctionParameters
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param value_function_parameters: how the function should be set up such as initial value
        """
        super().__init__(feature, value_function_parameters)
        self._w_lock: Optional[Lock] = None

    def attach_to_shared_weights(self, shared_w: SharedArrayWrapper):
        self._w_lock = shared_w.lock
        self.w = shared_w.array

    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        feature_matrix: np.ndarray = self._feature.get_matrix(state, actions)
        if self._w_lock:
            with self._w_lock:
                values_array: np.ndarray = self._feature.matrix_product(feature_matrix, self.w)
        else:
            values_array: np.ndarray = self._feature.matrix_product(feature_matrix, self.w)
        return values_array

    # def get_action_values3(self, state: State, actions: list[Action]) -> np.ndarray:
    #     values_array: np.ndarray = self._feature.get_dot_products(state, actions, self.w)
    #     return values_array

    def calc_value(self, feature_vector: np.ndarray) -> float:
        if self._w_lock:
            with self._w_lock:
                value: float = self._feature.dot_product(feature_vector, self.w)
        else:
            value: float = self._feature.dot_product(feature_vector, self.w)
        return value

    # Weights
    def update_weights(self, delta_w: np.ndarray):
        if self._w_lock:
            with self._w_lock:
                self.w += delta_w
        else:
            self.w += delta_w

    def update_weights_sparse(self, indices: np.ndarray, delta_w: float):
        if self._w_lock:
            with self._w_lock:
                self.w[indices] += delta_w
        else:
            self.w[indices] += delta_w