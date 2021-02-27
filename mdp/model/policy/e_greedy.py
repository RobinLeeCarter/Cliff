from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from mdp import common
if TYPE_CHECKING:
    from mdp.model import environment
    from mdp.model.policy import policy_
from mdp.model.policy import random, deterministic


class EGreedy(random.Random):
    def __init__(self, environment_: environment.Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self.greedy_policy: deterministic.Deterministic =\
            deterministic.Deterministic(self._environment, self._policy_parameters)
        self.epsilon = self._policy_parameters.epsilon

    def _get_action(self, state: environment.State) -> environment.Action:
        if common.rng.uniform() > self.epsilon:
            return self.greedy_policy[state]
        else:
            return random.Random._get_action(self, state)

    def __setitem__(self, state: environment.State, action: environment.Action):
        self.greedy_policy[state] = action

    @property
    def policy_for_display(self) -> policy_.Policy:
        return self.greedy_policy

    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        self.set_possible_actions(state_)
        non_greedy_p = self.epsilon * (1.0 / len(self.possible_actions))
        if action_ == self.greedy_policy[state_]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p
