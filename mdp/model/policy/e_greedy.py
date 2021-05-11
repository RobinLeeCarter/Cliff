from __future__ import annotations
from typing import TYPE_CHECKING

from mdp import common
if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.policy import policy
from mdp.model.policy import random, deterministic


class EGreedy(random.Random):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self.greedy_policy: deterministic.Deterministic =\
            deterministic.Deterministic(self._environment, self._policy_parameters)
        self.epsilon = self._policy_parameters.epsilon

    def _get_action(self, s: int) -> int:
        if common.rng.uniform() > self.epsilon:
            return self.greedy_policy[s]
        else:
            return random.Random._get_action(self, s)

    def __setitem__(self, s: int, a: int):
        self.greedy_policy[s] = a

    @property
    def linked_policy(self) -> policy.Policy:
        return self.greedy_policy

    def get_probability(self, s: int, a: int) -> float:
        non_greedy_p = self.epsilon * self._environment.one_over_possible_actions[s]
        if a == self.greedy_policy[s]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p
