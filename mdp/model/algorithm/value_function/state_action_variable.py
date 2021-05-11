from __future__ import annotations

from mdp.model.algorithm.value_function.state_action_function import StateActionFunction


class StateActionVariable(StateActionFunction):
    def initialize_values(self):
        self.matrix.fill(self._initial_value)

    def print_coverage_statistics(self):
        raise NotImplementedError
