from __future__ import annotations
from typing import TYPE_CHECKING

from mdp import common
if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment
    from mdp.model.policy import policy
from mdp.model.policy import random, deterministic


class EGreedy(random.Random):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self.greedy_policy: deterministic.Deterministic =\
            deterministic.Deterministic(self._environment, self._policy_parameters)
        self.epsilon = self._policy_parameters.epsilon

    def _get_action(self, state: State) -> Action:
        if common.rng.uniform() > self.epsilon:
            return self.greedy_policy[state]
        else:
            return random.Random._get_action(self, state)

    def __setitem__(self, state: State, action: Action):
        self.greedy_policy[state] = action

    @property
    def linked_policy(self) -> policy.Policy:
        return self.greedy_policy

    def get_probability(self, state: State, action: Action) -> float:
        self.set_possible_actions(state)
        non_greedy_p = self.epsilon * (1.0 / len(self.possible_actions))
        if action == self.greedy_policy[state]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p
