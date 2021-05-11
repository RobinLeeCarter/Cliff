from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.environment import Environment
from mdp.model.policy import policy


class Deterministic(policy.Policy):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        # self._action_for_state: dict[State, Action] = {}

        state_count = len(self._environment.states)
        # action_count = len(self._environment.actions)
        self.policy_vector: np.ndarray = np.zeros(shape=state_count, dtype=int)
        # self._policy_matrix: np.ndarray = np.zeros(shape=(state_count, action_count), dtype=float)

    # def reset(self):
    #     self._action_for_state: dict[State, Action] = {}
    #
    #     state_count = len(self._environment.states)
    #     action_count = len(self._environment.actions)
    #     self._policy_vector: np.ndarray = np.zeros(shape=state_count, dtype=int)
    #     self._policy_matrix: np.ndarray = np.zeros(shape=(state_count, action_count), dtype=float)

    def _get_action(self, s: int) -> int:
        return self.policy_vector[s]
        # state_index = self._environment.state_index[state]
        # return self._action_given_state[state_index]

    def __setitem__(self, s: int, a: int):
        # self._action_for_state[state] = action

        # s: int = self._environment.state_index[state]
        # a: int = self._environment.action_index[action]
        # prev_a = self._policy_vector[s]
        self.policy_vector[s] = a
        # self._policy_matrix[s, prev_a] = 0.0
        # self._policy_matrix[s, a] = 1.0

    # def set_policy_vector(self, policy_vector: np.ndarray, update_dict: bool = True):
    #     self._policy_vector = policy_vector
    #     if update_dict:
    #         for s, state in enumerate(self._environment.states):
    #             a = policy_vector[s]
    #             action: Action = self._environment.actions[a]
    #             self._action_for_state[state] = action
    #
    #         # prev_a = self._policy_vector[s]
    #         # self._policy_vector[s] = a
    #         # self._policy_matrix[s, prev_a] = 0.0
    #         # self._policy_matrix[s, a] = 1.0

    # def get_policy_vector(self) -> np.ndarray:
    #     return self._policy_vector

    # def get_policy_matrix(self) -> np.ndarray:
    #     return self._policy_matrix

        # state_count = len(self._environment.states)
        # action_count = len(self._environment.actions)
        # policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
        # for s in range(state_count):
        #     a = self._action_for_state_np[s]
        #     policy_matrix[s, a] = 1.0
        # return policy_matrix

    def get_probability(self, s: int, a: int) -> float:
        if a == self.policy_vector[s]:
            return 1.0
        else:
            return 0.0
