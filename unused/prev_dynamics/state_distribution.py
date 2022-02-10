from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Generator

import random

if TYPE_CHECKING:
    from mdp.model.environment import state
from mdp.model.environment.tabular.tabular_dynamics import reward_distribution


class StateDistribution:
    """normalised probability distribution p(s') of possible next states from Environment for one (s,a)"""
    def __init__(self):
        self._states: list[state.State] = []          # s'
        self._probabilities: list[float] = []   # p(s'|s,a)
        self._reward_distributions: list[reward_distribution.RewardDistribution] = []

        # duplication of lookup for performance
        self._state_lookup: dict[state.State, reward_distribution.RewardDistribution] = {}

        self._expected_reward: float = 0.0      # E[r|s,a] = Sum over r,s' of r.p(s',r|s,a)
        self._total_probability: float = 0.0    # should be 1.0

    @property
    def expected_reward(self) -> float:
        return self._expected_reward

    def add(self, state_: state.State, reward: float, probability: float):
        """add s', r, p(s',r|s,a)"""
        if state_ in self._states:
            index = self._states.index(state_)
            self._probabilities[index] += probability
            self._reward_distributions[index].add(reward, probability)
        else:
            self._states.append(state_)
            self._probabilities.append(probability)
            reward_distribution_ = reward_distribution.RewardDistribution()
            reward_distribution_.add(reward, probability)
            self._reward_distributions.append(reward_distribution_)
            self._state_lookup[state_] = reward_distribution_

        self._expected_reward += reward * probability
        self._total_probability += probability

    def get_state_probability(self, state_: state.State) -> float:
        """p(s'|s,a)"""
        reward_distribution_ = self._state_lookup.get(state_)
        if reward_distribution_:
            return reward_distribution_.state_probability
        else:
            return 0.0

    def get_state_reward_probability(self, state_: state.State, reward: float) -> float:
        """returns p(s',r|s,a)"""
        reward_distribution_ = self._state_lookup.get(state_)
        if reward_distribution_:
            return reward_distribution_.get_reward_probability(reward)
        else:
            return 0.0

    def next_states(self) -> Generator[(state.State, float, float), None, None]:
        """iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
        for state_, reward_distribution_ in zip(self._states, self._reward_distributions):
            yield state_, reward_distribution_.expected_reward, reward_distribution_.state_probability

    def states_and_rewards(self) -> Generator[(state.State, float, float), None, None]:
        """iterator for tuple(s', r, p(s',r|s,a))"""
        for state_, reward_distribution_ in zip(self._states, self._reward_distributions):
            for reward, probability in reward_distribution_.rewards():
                yield state_, reward, probability

    def draw(self) -> (state.State, float):
        """draw (s', r) from state and reward distributions p(s',r|s,a) """
        state_ = random.choices(self._states, weights=self._probabilities)[0]
        reward_distribution_ = self._state_lookup[state_]
        reward = reward_distribution_.draw()
        return state_, reward

