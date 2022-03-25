from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment


class StateActionVariable:
    def __init__(self,
                 environment: TabularEnvironment,
                 initial_value: float
                 ):
        self._environment: TabularEnvironment = environment
        self._initial_value: float = initial_value

        self.matrix: np.ndarray = np.empty(
            shape=(len(self._environment.states),
                   len(self._environment.actions)),
            dtype=float)

        self.initialize_values()

    def initialize_values(self):
        self.matrix.fill(self._initial_value)

    def __getitem__(self, s_a: tuple[int, int]) -> float:
        return self.matrix[s_a]

    def __setitem__(self, s_a: tuple[int, int], value: float):
        self.matrix[s_a] = value
