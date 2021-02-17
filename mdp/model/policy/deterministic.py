from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model import environment
from mdp.model.policy import policy_


class Deterministic(policy_.Policy):
    def __init__(self, environment_: environment.Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
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
