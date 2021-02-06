from typing import Optional, List

import numpy as np

import environment
import policy


class RandomPolicy(policy.Policy):
    # fully random
    def __init__(self, environment_: environment.Environment, rng: np.random.Generator):
        super().__init__(environment_)
        self.rng: np.random.Generator = rng

        # cache state and possible actions for get_probability to avoid doing it twice
        # self.state: Optional[environment.State] = None
        self.possible_actions: List[environment.Action] = []

    def get_action(self, state: environment.State) -> environment.Action:
        self.set_possible_actions(state)
        return self.rng.choice(self.possible_actions)

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        self.set_possible_actions(state_)
        return 1.0 / len(self.possible_actions)

    def set_possible_actions(self, state: environment.State):
        # if self.state is None or state != self.state:
        #       can't use cached version
        # self.state = state
        self.possible_actions = [action for action in self.environment.actions_for_state(state)]
        if not self.possible_actions:
            raise Exception(f"EGreedyPolicy state: {state} no possible actions")

    # pycharm is asking for this to be implemented even though it's not an abstract method, might be a pycharm bug
    def __setitem__(self, state: environment.State, action: environment.Action):
        super().__setitem__(state, action)
