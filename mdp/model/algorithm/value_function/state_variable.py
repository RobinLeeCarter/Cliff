from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.tabular_environment import TabularEnvironment


class StateVariable:
    def __init__(self,
                 environment_: TabularEnvironment,
                 initial_value: float
                 ):
        self._environment: TabularEnvironment = environment_
        self._initial_value: float = initial_value

        self.vector: np.ndarray = np.empty(
            shape=len(self._environment.states),
            dtype=float)

        self.initialize_values()

    def initialize_values(self):
        self.vector.fill(self._initial_value)

    def __getitem__(self, s: int) -> float:
        return self.vector[s]

    def __setitem__(self, s: int, value: float):
        self.vector[s] = value

    def print_all_values(self):
        print("Variable vector ...")
        print(self.vector)
