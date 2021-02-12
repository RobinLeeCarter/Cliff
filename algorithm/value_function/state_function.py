from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import environment


class StateFunction:
    def __init__(self,
                 environment_: environment.Environment,
                 initial_v_value: float
                 ):
        self._environment: environment.Environment = environment_
        self.initial_v_value: float = initial_v_value

        self._shape = self._environment.states_shape
        self._values: np.ndarray = np.empty(shape=self._shape, dtype=float)

    def initialize_values(self):
        for state_ in self._environment.states():
            if state_.is_terminal:
                self._values[state_.index] = 0.0
            else:
                self._values[state_.index] = self.initial_v_value

    def __getitem__(self, state: environment.State) -> float:
        if state.is_terminal:
            return 0.0
        else:
            return self._values[state.index]

    def __setitem__(self, state: environment.State, value: float):
        self._values[state.index] = value

    def print_coverage_statistics(self):
        v_size = self._values.size
        v_non_zero = np.count_nonzero(self._values)
        percent_non_zero = 100.0 * v_non_zero / v_size
        print(f"v_size: {v_size}\tv_non_zero: {v_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")
