from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.model import environment
from mdp.model.policy import policy_


class Deterministic(policy_.Policy):
    def __init__(self, environment_: environment.Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self._action_for_state: dict[environment.State, environment.Action] = {}
        # self._action_given_state: np.ndarray = np.empty(
        #     shape=len(environment_.states),
        #     dtype=environment_.action_type)

    def _get_action(self, state: environment.State) -> environment.Action:
        return self._action_for_state[state]
        # state_index = self._environment.state_index[state]
        # return self._action_given_state[state_index]

    def __setitem__(self, state: environment.State, action: environment.Action):
        self._action_for_state[state] = action
        # state_index = self._environment.state_index[state]
        # self._action_given_state[state_index] = action

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        if action_ == self[state_]:
            return 1.0
        else:
            return 0.0
