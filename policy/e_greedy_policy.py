from __future__ import annotations
from typing import TYPE_CHECKING

import common

from policy import random_policy
if TYPE_CHECKING:
    import environment
    from policy import deterministic_policy


class EGreedyPolicy(random_policy.RandomPolicy):
    def __init__(self, environment_: environment.Environment,
                 greedy_policy: deterministic_policy.DeterministicPolicy, epsilon: float = 0.1):
        super().__init__(environment_)
        self.greedy_policy: deterministic_policy.DeterministicPolicy = greedy_policy
        self.epsilon = epsilon

    def get_action(self, state: environment.State) -> environment.Action:
        if common.rng.uniform() > self.epsilon:
            return self.greedy_policy[state]
        else:
            return random_policy.RandomPolicy.get_action(self, state)

    def __setitem__(self, state: environment.State, action: environment.Action):
        self.greedy_policy[state] = action

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        self.set_possible_actions(state_)
        non_greedy_p = self.epsilon * (1.0 / len(self.possible_actions))
        if action_ == self.greedy_policy[state_]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p
