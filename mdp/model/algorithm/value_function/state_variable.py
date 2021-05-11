from __future__ import annotations

from mdp.model.algorithm.value_function.state_function import StateFunction


class StateVariable(StateFunction):
    def initialize_values(self):
        for s in range(len(self._environment.states)):
            self.vector[s] = self._initial_value

    def print_all_values(self):
        print("Variable vector ...")
        print(self.vector)

    def print_coverage_statistics(self):
        raise NotImplementedError
