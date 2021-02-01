import itertools

import numpy as np

import constants
import environment


class StateActionFunction:
    def __init__(self,
                 environment_: environment.Environment
                 ):
        self._environment: environment.Environment = environment_
        self._shape = self._environment.states_shape + self._environment.actions_shape
        self._values: np.ndarray = np.empty(shape=self._shape, dtype=float)
        # self._actions_slice is used by argmax_over_actions so cached here
        all_slice: slice = np.s_[:]
        self._actions_slice: tuple[slice] = tuple(itertools.repeat(all_slice, len(self._environment.actions_shape)))

        self._initialize_q()

    def _initialize_q(self):
        # incompatible actions must never be selected
        self._values.fill(np.NINF)
        # so that a successful trajectory is always better
        for state_ in self._environment.states():
            for action_ in self._environment.actions_for_state(state_):
                q_index = state_.index + action_.index
                if state_.is_terminal:
                    self._values[q_index] = 0.0
                else:
                    self._values[q_index] = constants.INITIAL_Q_VALUE

    def __getitem__(self, state_action: tuple[environment.State, environment.Action]) -> float:
        state, action = state_action
        if state.is_terminal or action is None:
            return 0.0
        else:
            return self._values[state.index + action.index]

    def __setitem__(self, state_action: tuple[environment.State, environment.Action], value: float):
        state, action = state_action
        self._values[state.index + action.index] = value

    # def get_value(self, state: environment.State, action: environment.Action) -> float:
    #     return self._q_values[state.index + action.index]
    #
    # def set_value(self, state: environment.State, action: environment.Action, value: float):
    #     self._q_values[state.index + action.index] = value

    def argmax_over_actions(self, state: environment.State) -> environment.Action:
        """set target_policy to argmax over a of Q breaking ties consistently"""
        # state_index = self.get_index_from_state(state_)
        # print(f"state_index {state_index}")
        q_slice = state.index + self._actions_slice
        q_state: np.ndarray = self._values[q_slice]
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

        # best_flat_index_np is just an int but can't be typed as such
        best_flat_index_np: np.ndarray = np.argmax(q_state)
        # best_index_np is actually tuple[np.int64] but can't be typed as such
        best_index_np: tuple[np.ndarray] = np.unravel_index(best_flat_index_np, shape=q_state.shape)
        # assert np.isscalar(best_index_np[0]) - could assert but don't need to
        best_index: tuple = tuple(int(i) for i in best_index_np)

        # best_index_np: tuple = best_index_tuple_array[0][0]
        # print(f"best_index_np {best_index_np}")
        best_action = environment.Actions.get_action_from_index(best_index)
        # best_action = self.get_action_from_index(best_index_np)
        # print(f"best_action {best_action}")
        return best_action

    def print_coverage_statistics(self):
        q_size = self._values.size
        q_non_zero = np.count_nonzero(self._values)
        percent_non_zero = 100.0 * q_non_zero / q_size
        print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")
