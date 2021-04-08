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
        # self._action_given_state: np.ndarray = np.empty(
        #     shape=len(environment_.states),
        #     dtype=environment_.action_type)

    def _get_action(self, state: State) -> Action:
        return self._action_for_state[state]
        # state_index = self._environment.state_index[state]
        # return self._action_given_state[state_index]

    def __setitem__(self, state: State, action: Action):
        self._action_for_state[state] = action
        # state_index = self._environment.state_index[state]
        # self._action_given_state[state_index] = action

    def get_probability(self, state: State, action: Action) -> float:
        if action == self[state]:
            return 1.0
        else:
            return 0.0

    def set_policy_vector(self, policy_vector: np.ndarray):
        for s, state in enumerate(self._environment.states):
            a = policy_vector[s]
            action: Action = self._environment.actions[a]
            self._action_for_state[state] = action
