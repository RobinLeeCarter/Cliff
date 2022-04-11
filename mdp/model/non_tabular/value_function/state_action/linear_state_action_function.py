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
        self.delta_w: Optional[np.ndarray] = None
        if value_function_parameters.requires_delta_w:
            self.delta_w: np.ndarray = np.zeros_like(self.w)

    def has_sparse_feature(self) -> bool:
        return self._feature.is_sparse

    def __getitem__(self, state_action: tuple[State, Action]) -> float:
        state, action = state_action
        if state.is_terminal:
            return 0.0
        else:
            self._feature.set_state_action(state, action)
            return self._feature.dot_product_full_vector(self.w)

    def get_gradient(self, state: State, action: Action) -> np.ndarray:
        # for linear case the gradient doesn't depend on the state and action
        assert action is not None
        return self._feature.get_vector()

    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        values: list[float] = []
        # set state just once
        self._feature.set_state(state)
        for action in actions:
            self._feature.set_action(action)
            # will calculate vector and then use it
            value = self._feature.dot_product_full_vector(self.w)
            values.append(value)
        return np.array(values)

    def update_weights(self, delta_w: np.ndarray):
        self.w += delta_w

    def update_weights_sparse(self, indices: np.ndarray, delta_w: float):
        self.w[indices] += delta_w

    # Delta weights
    def reset_delta_w(self):
        # non_zero_w: int = np.count_nonzero(self.w)
        # print(f"{non_zero_w=}")
        self.delta_w.fill(0.0)

    # def set_delta_weights(self, delta_w: np.ndarray):
    #     self.delta_w = delta_w

    def update_delta_weights(self, delta_w: np.ndarray):
        self.delta_w += delta_w

    def update_delta_weights_sparse(self, indices: np.ndarray, delta_w: float):
        self.delta_w[indices] += delta_w

    def get_delta_weights(self) -> np.ndarray:
        return self.delta_w

    def apply_delta_weights(self):
        self.w += self.delta_w
