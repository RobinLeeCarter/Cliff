from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp import common
    from mdp.model import environment
from mdp.model.policy import policy


class NoPolicy(policy.Policy):
    def __init__(self, environment_: environment.Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        # actions: list[environment.Action] = [action for action in environment_.actions()]
        # self.action = actions[0]    # always pick first action (presumably stationary)

    def _get_action(self, state: environment.State) -> Optional[environment.Action]:
        return None

    def __setitem__(self, state: environment.State, action: environment.Action):
        super().__setitem__(state, action)

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        if action_ is None:
            return 1.0
        else:
            return 0.0
