from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment_tabular import EnvironmentTabular


class StateFunction:
    def __init__(self,
                 environment_: EnvironmentTabular,
                 initial_value: float
                 ):
        self._environment: EnvironmentTabular = environment_
        self._initial_value: float = initial_value

        self.vector: np.ndarray = np.empty(
            shape=len(self._environment.states),
            dtype=float)

        self.initialize_values()

    def initialize_values(self):
        for s in range(len(self._environment.states)):
            if self._environment.is_terminal[s]:
                self.vector[s] = 0.0
            else:
                self.vector[s] = self._initial_value

    def __getitem__(self, s: int) -> float:
        return self.vector[s]

    def __setitem__(self, s: int, value: float):
        self.vector[s] = value

    def print_all_values(self):
        print("V.vector ...")
        print(self.vector)

    def print_coverage_statistics(self):
        v_size = self.vector.size
        v_non_zero = np.count_nonzero(self.vector)
        percent_non_zero = 100.0 * v_non_zero / v_size
        print(f"v_size: {v_size}\tv_non_zero: {v_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")
