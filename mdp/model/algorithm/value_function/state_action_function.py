from __future__ import annotations
from typing import TYPE_CHECKING
# import itertools

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment


class StateActionFunction:
    def __init__(self,
                 environment: Environment,
                 initial_q_value: float
                 ):
        self._environment: Environment = environment
        self._initial_q_value: float = initial_q_value

        self._values: np.ndarray = np.empty(
            shape=(len(self._environment.states),
                   len(self._environment.actions)),
            dtype=float)

        self.initialize_values()

    def initialize_values(self):
        # incompatible actions must never be selected
        self._values.fill(np.NINF)
        # so that a successful trajectory is always better
        for state_ in self._environment.states:
            for action_ in self._environment.actions_for_state[state_]:
                state_action_index = self._environment.state_action_index(state_, action_)
                if state_.is_terminal:
                    self._values[state_action_index] = 0.0
                else:
                    self._values[state_action_index] = self._initial_q_value

    def __getitem__(self, state_action: tuple[State, Action]) -> float:
        state, action = state_action
        if state.is_terminal or action is None:
            return 0.0
        else:
            state_action_index = self._environment.state_action_index(state, action)
            return self._values[state_action_index]

    def __setitem__(self, state_action: tuple[State, Action], value: float):
        state, action = state_action
        state_action_index = self._environment.state_action_index(state, action)
        self._values[state_action_index] = value

    def argmax_over_actions(self, state: State) -> Action:
        """set target_policy to argmax over a of Q breaking ties consistently"""
        # state_index = self.get_index_from_state(state_)
        # print(f"state_index {state_index}")
        state_index = self._environment.state_index[state]
        # q_slice = state.index + self._actions_slice
        q_state: np.ndarray = self._values[state_index, :]
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
        # # best_flat_index_np is just an int but can'_t be typed as such
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
        best_flat_index: int = int(np.argmax(q_state))
        best_action = self._environment.actions[best_flat_index]

        return best_action

    def max_over_actions(self, state: State) -> float:
        """max_over_a Q[state, a]"""
        state_index = self._environment.state_index[state]
        q_state: np.ndarray = self._values[state_index, :]
        return np.max(q_state)

    def print_coverage_statistics(self):
        q_size = self._values.size
        q_non_zero = np.count_nonzero(self._values)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")
