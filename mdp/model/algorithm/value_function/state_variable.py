from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model import environment


class StateVariable:
    def __init__(self,
                 environment_: environment.Environment,
                 initial_value: float
                 ):
        self._environment: environment.Environment = environment_
        self._initial_value: float = initial_value

        self._values: np.ndarray = np.empty(
            shape=len(self._environment.states),
            dtype=float)

        self.initialize_values()

    def initialize_values(self):
        for state_ in self._environment.states:
            state_index = self._environment.state_index[state_]
            self._values[state_index] = self._initial_value

    def __getitem__(self, state: environment.State) -> float:
        state_index = self._environment.state_index[state]
        return self._values[state_index]

    def __setitem__(self, state: environment.State, value: float):
        state_index = self._environment.state_index[state]
        self._values[state_index] = value