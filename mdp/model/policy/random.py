from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment
from mdp import common
from mdp.model.policy import policy


class Random(policy.Policy):
    # fully random
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)

        # cache state and possible actions for get_probability to avoid doing it twice
        # self.state: Optional[environment.State] = None
        self.possible_actions: List[Action] = []

    def _get_action(self, state: State) -> Action:
        self.set_possible_actions(state)
        return common.rng.choice(self.possible_actions)

    def get_probability(self, state: State, action: Action) -> float:
        self.set_possible_actions(state)
        return 1.0 / len(self.possible_actions)

    def set_possible_actions(self, state: State):
        # if self.state is None or state != self.state:
        #       can'_t use cached version
        # self.state = state
        self.possible_actions = [action for action in self._environment.actions_for_state(state)]
        if not self.possible_actions:
            raise Exception(f"Random state: {state} no possible actions")

    # pycharm is asking for this to be implemented even though it's not an abstract method, might be a pycharm bug
    def __setitem__(self, state: State, action: Action):
        super().__setitem__(state, action)
