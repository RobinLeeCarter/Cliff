from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from mdp import common
from unused.unused_feature_weights import FeatureWeights

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
        self.weights: FeatureWeights = FeatureWeights(feature, self._initial_value)
        # self.size: int = self._feature.max_size
        # weights
        # self.w: np.ndarray = np.full(shape=self.size, fill_value=self._initial_value, dtype=float)

    def has_sparse_feature(self) -> bool:
        return self._feature.is_sparse

    def __getitem__(self, state_action: tuple[State, Action]) -> float:
        state, action = state_action
        if state.is_terminal:
            return 0.0
        else:
            self._feature.set_state_action(state, action)
            return self._feature.dot_product_full_vector(self.weights.w)

    def get_gradient(self, state: State, action: Action) -> np.ndarray:
        return self._feature.get_vector()

    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        values: list[float] = []
        # set state just once
        self._feature.set_state(state)
        for action in actions:
            self._feature.set_action(action)
            # will calculate vector and then use it
            value = self._feature.dot_product_full_vector(self.weights.w)
            values.append(value)
        return np.array(values)

    def update_weights(self, delta_w: np.ndarray):
        self.weights.update_weights(delta_w)
        # self.w += delta_w

    def update_weights_sparse(self, indices: np.ndarray, delta_w: float):
        self.weights.update_weights_sparse(indices, delta_w)
        # self.w[indices] += delta_w
