from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from policy import policy
if TYPE_CHECKING:
    import environment


class DeterministicPolicy(policy.Policy):
    def __init__(self, environment_: environment.Environment):
        super().__init__(environment_)
        self._action_given_state: np.ndarray = np.empty(shape=environment_.states_shape, dtype=environment_.action_type)

    def get_action(self, state: environment.State) -> environment.Action:
        return self._action_given_state[state.index]

    def __setitem__(self, state: environment.State, action: environment.Action):
        self._action_given_state[state.index] = action

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        if action_ == self[state_]:
            return 1.0
        else:
            return 0.0
