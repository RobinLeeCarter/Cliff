from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment


class StateActionFunction:
    def __init__(self,
                 environment: Environment,
                 initial_value: float
                 ):
        self._environment: Environment = environment
        self._initial_value: float = initial_value

        self.matrix: np.ndarray = np.empty(
            shape=(len(self._environment.states),
                   len(self._environment.actions)),
            dtype=float)
        self.max: np.ndarray = np.empty(shape=len(self._environment.states), dtype=float)
        self.argmax: np.ndarray = np.empty(shape=len(self._environment.states), dtype=int)

        self.initialize_values()

    def initialize_values(self):
        for s in range(len(self._environment.states)):
            for a in range(len(self._environment.actions)):
                if self._environment.is_terminal[s]:
                    self.matrix[s, a] = 0.0
                else:
                    if self._environment.s_a_compatibility[s, a]:
                        self.matrix[s, a] = self._initial_value
                    else:
                        # incompatible actions must never be selected
                        self.matrix[s, a] = np.NINF

        self.max = np.max(self.matrix, axis=1)
        self.argmax = np.argmax(self.matrix, axis=1)

    def __getitem__(self, s_a: tuple[int, int]) -> float:
        return self.matrix[s_a]

    def __setitem__(self, s_a: tuple[int, int], value: float):
        # print(f"s_a: {s_a} \tvalue: {value}")
        self.matrix[s_a] = value

        s, a = s_a
        current_max = self.max[s]
        if value > current_max:
            self.max[s] = value
            self.argmax[s] = a
        elif value == current_max:
            if a < self.argmax[s]:   # so consistent
                self.argmax[s] = a
        else:
            if a == self.argmax[s]:
                new_argmax = np.argmax(self.matrix[s, :])
                self.argmax[s] = new_argmax
                self.max[s] = self.matrix[s, new_argmax]

    def argmax_over_actions(self, s: int) -> int:
        """argmax over a of Q breaking ties consistently"""
        return self.argmax[s]
        # return int(np.argmax(self.matrix[s, :]))

    def max_over_actions(self, s: int) -> float:
        """max_over_a Q[state, a]"""
        return self.max[s]
        # return np.max(self.matrix[s, :])

    def print_coverage_statistics(self):
        q_size = self.matrix.size
        q_non_zero = np.count_nonzero(self.matrix)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")

    # state_index = self.get_index_from_state(state_)
    # print(f"state_index {state_index}")

    # state_index = self._environment.state_index[state]

    # q_slice = state.index + self._actions_slice

    # q_state: np.ndarray = self.values[state_index, :]

    # print(f"q_state.shape {q_state.shape}")

    # argmax
    # best_q: float = np.max(q_state)
    # # print(f"best_q {best_q}")
    # best_q_bool: np.ndarray = (q_state == best_q)
    # # print(f"best_q_bool.shape {best_q_bool.shape}")
    # best_flat_indexes: np.ndarray = np.flatnonzero(best_q_bool)
    # best_flat_indexes: np.ndarray = np.argmax(q_state)
    # consistent_best_flat_index: int = best_flat_indexes[0]
    # consistent_best_flat_index: int = np.argmax(q_state)
    # print(f"consistent_best_flat_index {consistent_best_flat_index}")

    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # In case of multiple occurrences of the maximum values,
    # the indices corresponding to the first occurrence are returned

    # 2D version
    # # best_flat_index_np is just an int but can't be typed as such
    # best_flat_index_np: np.ndarray = np.argmax(q_state)
    # # best_index_np is actually tuple[np.int64] but can'_t be typed as such
    # best_index_np: tuple[np.ndarray] = np.unravel_index(best_flat_index_np, shape=q_state.shape)
    # # assert np.isscalar(best_index_np[0]) - could assert but don'_t need to
    # best_index: tuple = tuple(int(i) for i in best_index_np)
    #
    # # best_index_np: tuple = best_index_tuple_array[0][0]
    # # print(f"best_index_np {best_index_np}")
    # best_action = self._environment.get_action_from_index(best_index)
    # # best_action = self.get_action_from_index(best_index_np)
    # # print(f"best_action {best_action}")

    # 1D version
    # best_flat_index: int = int(np.argmax(q_state))
    # best_action = self._environment.actions[best_flat_index]
    #
    # return best_action
