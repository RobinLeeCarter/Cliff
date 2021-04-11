from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment
from mdp.model.policy import policy


class Deterministic(policy.Policy):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self._action_for_state: dict[State, Action] = {}

        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        self._action_for_state_np: np.ndarray = np.zeros(shape=state_count, dtype=int)
        self._policy_matrix: np.ndarray = np.zeros(shape=(state_count, action_count), dtype=float)

    def _get_action(self, state: State) -> Action:
        return self._action_for_state[state]
        # state_index = self._environment.state_index[state]
        # return self._action_given_state[state_index]

    def __setitem__(self, state: State, action: Action):
        self._action_for_state[state] = action

        s: int = self._environment.state_index[state]
        a: int = self._environment.action_index[action]
        prev_a = self._action_for_state_np[s]
        self._action_for_state_np[s] = a
        self._policy_matrix[s, prev_a] = 0.0
        self._policy_matrix[s, a] = 1.0
        # state_index = self._environment.state_index[state]
        # self._action_given_state[state_index] = action

    def set_policy_vector(self, policy_vector: np.ndarray):
        for s, state in enumerate(self._environment.states):
            a = policy_vector[s]
            action: Action = self._environment.actions[a]
            self._action_for_state[state] = action

            prev_a = self._action_for_state_np[s]
            self._action_for_state_np[s] = a
            self._policy_matrix[s, prev_a] = 0.0
            self._policy_matrix[s, a] = 1.0

    def get_policy_matrix(self) -> np.ndarray:
        return self._policy_matrix

        # state_count = len(self._environment.states)
        # action_count = len(self._environment.actions)
        # policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
        # for s in range(state_count):
        #     a = self._action_for_state_np[s]
        #     policy_matrix[s, a] = 1.0
        # return policy_matrix

    def get_probability(self, state: State, action: Action) -> float:
        if action == self[state]:
            return 1.0
        else:
            return 0.0
