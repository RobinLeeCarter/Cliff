from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Generator

import random
# import numpy as np

from mdp import common


@dataclass(frozen=True)
class State:
    name: str


@dataclass(frozen=True)
class Action:
    name: str


@dataclass(frozen=True)
class Response:
    """(s',r) = (new_state, reward)"""
    state: State
    reward: Optional[float]


class RewardDistribution:
    """un-normalised probability distribution p(r) of possible rewards from Environment for one (s,a,s')"""
    def __init__(self):
        # using a list rather than a dictionary as 20-30% faster for draw() which is called with high frequency
        """
        self._rewards: r
        self._probabilities: p(s',r|s,a)
        self.expected_reward: E[r|s,a,s']
        self.state_probability: p(s'|s,a)
        """
        self._rewards: list[float] = []         # r
        self._probabilities: list[float] = []   # p(s',r|s,a)

        self.expected_reward: float = 0.0       # E[r|s,a,s']
        self.state_probability: float = 0.0     # p(s'|s,a)

    def add(self, reward: float, probability: float):
        if reward in self._rewards:
            index = self._rewards.index(reward)
            self._probabilities[index] += probability
        else:
            self._rewards.append(reward)
            self._probabilities.append(probability)

        self.expected_reward += reward * probability
        self.state_probability += probability

    def get_reward_probability(self, reward: float) -> float:
        """get specific p(s',r|s,a)
        unlikely as usually iterating over so slow is acceptable
        may not match as matching on a float"""
        if reward in self._rewards:
            index = self._rewards.index(reward)
            return self._probabilities[index]
        else:
            return 0.0

    def __iter__(self) -> Generator[(float, float), None, None]:
        """r, p(s',r|s,a)"""
        for reward, probability in zip(self._rewards, self._probabilities):
            yield reward, probability

    def draw(self) -> float:
        """draw r from rewards distribution"""
        return random.choices(self._rewards, weights=self._probabilities)[0]


class StateDistribution:
    """normalised probability distribution p(s') of possible next states from Environment for one (s,a)"""
    def __init__(self):
        self._states: list[State] = []          # s'
        self._probabilities: list[float] = []   # p(s'|s,a)
        self._reward_distributions: list[RewardDistribution] = []

        # duplication of lookup for performance
        self._state_lookup: dict[State, RewardDistribution] = {}

        self._expected_reward: float = 0.0      # E[r|s,a]
        self._total_probability: float = 0.0    # should be 1.0

    @property
    def expected_reward(self) -> float:
        return self._expected_reward

    def add(self, state: State, reward: float, probability: float):
        """add s', r, p(s',r|s,a)"""
        if state in self._states:
            index = self._states.index(state)
            self._probabilities[index] += probability
            self._reward_distributions[index].add(reward, probability)
        else:
            self._states.append(state)
            self._probabilities.append(probability)
            reward_distribution = RewardDistribution()
            reward_distribution.add(reward, probability)
            self._reward_distributions.append(reward_distribution)
            self._state_lookup[state] = reward_distribution

        self._expected_reward += reward * probability
        self._total_probability += probability

    def get_state_probability(self, state: State) -> float:
        """p(s'|s,a)"""
        reward_distribution = self._state_lookup.get(state)
        if reward_distribution:
            return reward_distribution.state_probability
        else:
            return 0.0

    def get_state_reward_probability(self, state: State, reward: float) -> float:
        """returns p(s',r|s,a)"""
        reward_distribution = self._state_lookup.get(state)
        if reward_distribution:
            return reward_distribution.get_reward_probability(reward)
        else:
            return 0.0

    def __iter__(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
        for state, reward_distribution in zip(self._states, self._reward_distributions):
            yield state, reward_distribution.expected_reward, reward_distribution.state_probability

    def state_reward(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple(s', r, p(s',r|s,a))"""
        for state, reward_distribution in zip(self._states, self._reward_distributions):
            for reward, probability in reward_distribution:
                yield state, reward, probability

    def draw(self) -> (State, float):
        """draw (s', r) from state and reward distributions p(s',r|s,a) """
        state = random.choices(self._states, weights=self._probabilities)[0]
        reward_distribution = self._state_lookup[state]
        reward = reward_distribution.draw()
        return state, reward


class Dynamics:
    """p(s',r|s,a)"""
    def __init__(self):
        self._state_distributions: dict[tuple[State, Action], StateDistribution] = {}

    def add(self, state: State, action: Action, new_state: State, reward: float, probability: float):
        """add s, a, s', r, p(s',r|s,a)"""
        state_action: tuple[State, Action] = (state, action)
        state_distribution = self._state_distributions.get(state_action)
        if state_distribution:
            state_distribution.add(new_state, reward, probability)
        else:
            state_distribution = StateDistribution()
            state_distribution.add(new_state, reward, probability)
            self._state_distributions[state_action] = state_distribution

    def draw(self, state: State, action: Action) -> (State, float):
        """pass (s,a) get (s',r)"""
        state_action: tuple[State, Action] = (state, action)
        state_distribution = self._state_distributions[state_action]
        return state_distribution.draw()

    def states(self, state: State, action: Action) -> Generator[(State, float, float), None, None]:
        """pass (s,a) iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
        state_action: tuple[State, Action] = (state, action)
        state_distribution = self._state_distributions.get(state_action)
        yield from state_distribution

    def state_rewards(self, state: State, action: Action) -> Generator[(State, float, float), None, None]:
        """pass (s,a) iterator for tuple(s', r, p(s',r|s,a))"""
        state_action: tuple[State, Action] = (state, action)
        state_distribution = self._state_distributions.get(state_action)
        yield from state_distribution.state_reward()


# test code

dist = RewardDistribution()
dist.add(reward=10.0, probability=0.1)
dist.add(reward=20.0, probability=0.2)
dist.add(reward=30.0, probability=0.3)
dist.add(reward=40.0, probability=0.4)

for reward_, probability_ in dist:
    print(reward_, probability_)

print(dist.draw())
print(dist.draw())
print(dist.draw())
print(dist.draw())

hello_state = State(name="hello")
left_action = Action(name="left")
right_action = Action(name="right")
goodbye_state = State(name="goodbye")
response_1 = Response(hello_state, 100.0)
response_2 = Response(goodbye_state, 1.0)

