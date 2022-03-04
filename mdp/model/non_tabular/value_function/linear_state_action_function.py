from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.feature.feature import Feature
from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class LinearStateActionFunction(StateActionFunction[State, Action]):
    def __init__(self,
                 feature: Feature,
                 initial_value: float
                 ):
        """
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param initial_value: initial value for the weights (else zero)
        """
        self.feature: Feature = feature
        self.initial_value: float = initial_value

        self.size: int = self.feature.max_size
        # weights
        self.w: np.ndarray = np.full(shape=self.size, fill_value=initial_value, dtype=float)

    def __getitem__(self, state: State, action: Action) -> float:
        self.feature.set_state_action(state, action)
        return self.feature.dot_product_full_vector(self.w)

    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        values: list[float] = []
        # set state just once
        self.feature.set_state(state)
        for action in actions:
            self.feature.set_action(action)
            # will calculate vector and then use it
            value = self.feature.dot_product_full_vector(self.w)
            values.append(value)
        return np.array(values)
