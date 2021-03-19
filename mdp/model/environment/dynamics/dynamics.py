from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment import State, Action

from mdp.model.environment import StateAction
from mdp.model.environment.dynamics.state_probability import StateProbability
# from mdp.model.environment.dynamics.next_state_distribution import NextStateDistribution


class Dynamics:
    def __init__(self):
        self.expected_reward_dict: dict[StateAction, float] = {}
        self.next_state_dist_dict: dict[StateAction, list[StateProbability]] = {}

    def set_expected_reward(self, state: State, action: Action, expected_reward: float):
        state_action = StateAction(state, action)
        self.expected_reward_dict[state_action] = expected_reward

    def get_expected_reward(self, state: State, action: Action) -> float:
        state_action = StateAction(state, action)
        return self.expected_reward_dict[state_action]

    def set_next_state_probability(self, state: State, action: Action,
                                   next_state: State, probability: float):
        state_action = StateAction(state, action)

        next_state_distribution = self.next_state_dist_dict.get(state_action)
        if not next_state_distribution:
            next_state_distribution: list[StateProbability] = []
            self.next_state_dist_dict[state_action] = next_state_distribution

        state_probability = StateProbability(next_state, probability)
        next_state_distribution.append(state_probability)

    def get_next_state_distribution(self, state: State, action: Action) -> list[StateProbability]:
        state_action = StateAction(state, action)
        return self.next_state_dist_dict[state_action]
