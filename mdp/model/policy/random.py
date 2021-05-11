from __future__ import annotations
# from typing import TYPE_CHECKING

import numpy as np

# if TYPE_CHECKING:
#     from mdp.model.environment.environment import Environment
from mdp import common
from mdp.model.policy import policy


class Random(policy.Policy):
    # fully random
    # def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
    #     super().__init__(environment_, policy_parameters)
    #     # cache state and possible actions for get_probability to avoid doing it twice
    #     self.state: Optional[environment.State] = None
    #     self.possible_actions: List[Action] = []

    def _get_action(self, s: int) -> int:
        return common.rng.choice(
            np.flatnonzero(
                self._environment.s_a_compatibility[s, :]
            )
        )

    def get_probability(self, s: int, a: int) -> float:
        if self._environment.s_a_compatibility[s, a]:
            return self._environment.one_over_possible_actions[s]
        else:
            return 0.0
        # self.set_possible_actions(state)
        # possible_actions: int = np.count_nonzero(self._environment.s_a_compatibility[s, :])
        # return 1.0 / possible_actions

    # def set_possible_actions(self, state: State):
    #     # if self.state is None or state != self.state:
    #     #       can'_t use cached version
    #     # self.state = state
    #     self.possible_actions = self._environment.actions_for_state[state]
    #     if not self.possible_actions:
    #         raise Exception(f"Random state: {state} no possible actions")

    # pycharm is asking for this to be implemented even though it's not an abstract method, might be a pycharm bug
    def __setitem__(self, s: int, a: int):
        super().__setitem__(s, a)
