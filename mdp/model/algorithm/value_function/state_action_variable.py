from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment


class StateActionVariable:
    def __init__(self,
                 environment_: Environment,
                 initial_value: float = 0.0
                 ):
        self._environment: Environment = environment_
        self._initial_value: float = initial_value

        self._values: np.ndarray = np.empty(
            shape=(len(self._environment.states),
                   len(self._environment.actions)),
            dtype=float)

        self.initialize_values()

    def initialize_values(self):
        for state_ in self._environment.states:
            for action_ in self._environment.actions_for_state[state_]:
                state_action_index = self._environment.state_action_index(state_, action_)
                self._values[state_action_index] = self._initial_value

    def __getitem__(self, state_action: tuple[State, Action]) -> float:
        state, action = state_action
        state_action_index = self._environment.state_action_index(state, action)
        return self._values[state_action_index]

    def __setitem__(self, state_action: tuple[State, Action], value: float):
        state, action = state_action
        state_action_index = self._environment.state_action_index(state, action)
        self._values[state_action_index] = value
