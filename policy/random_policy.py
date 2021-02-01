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
        self.state: Optional[environment.State] = None
        self.possible_actions: List[environment.Action] = []

    def __getitem__(self, state: environment.State) -> environment.Action:
        self.set_possible_actions(state)
        return self.rng.choice(self.possible_actions)

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        self.set_possible_actions(state_)
        return 1.0 / len(self.possible_actions)

    def set_possible_actions(self, state: environment.State):
        if self.state is None or state != self.state:
            # can't use cached version
            self.state = state
            self.possible_actions = [action for action in self.environment.actions_for_state(state)]
            if not self.possible_actions:
                raise Exception(f"EGreedyPolicy state: {state} no possible actions")
