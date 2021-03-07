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

    def __iter__(self) -> Generator[(float, float), None, None]:
        for reward, probability in zip(self._rewards, self._probabilities):
            yield reward, probability

    def draw(self) -> float:
        """draw a reward from reward distribution"""
        return random.choices(self._rewards, weights=self._probabilities)[0]


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
            return 0

    def __iter__(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple (s', E[r|s,a,s'], p(s'|s,a)) """
        for state, reward_distribution in zip(self._states, self._reward_distributions):
            yield state, reward_distribution.expected_reward, reward_distribution.state_probability

    def responses(self) -> Generator[(Response, float), None, None]:
        """iterator for tuple (s', r, p(s',r|s,a) ) """
        for state, reward_distribution in zip(self._states, self._reward_distributions):
            for reward, probability in reward_distribution:
                yield Response(state, reward), probability

    def draw(self) -> Response:
        """draw a response:(state, reward) from state and reward distributions a.k.a p(s',r|s,a) """
        state = random.choices(self._states, weights=self._probabilities)[0]
        reward_distribution = self._state_lookup[state]
        reward = reward_distribution.draw()
        return Response(state, reward)

    # def get_state_expected_rewards(self, new_state: State) -> float:
    #     return self._distribution[new_state].expected_reward

    # def draw(self) -> Response:
    #     pass


hello_state = State(name="hello")
left_action = Action(name="left")
right_action = Action(name="right")
goodbye_state = State(name="goodbye")
response_1 = Response(hello_state, 100.0)
response_2 = Response(goodbye_state, 1.0)


class ResponseDistribution:
    """probability distribution p(s',r) of possible responses from Environment for one (s,a)
    (new_state, reward): probability"""
    def __init__(self):
        # or dict[Response, float] but reward is float so unwise
        self._distribution: dict[Response, float] = {}
        # self._distribution: list[(Response, float)] = []

    def __getitem__(self, response_: Response) -> float:
        return self._distribution[response_]    # may or may not work as Response key contains float

    def __setitem__(self, response_: Response, probability: float):
        self._distribution[response_] = probability
        # self._distribution.append((response_, probability_))

    def __iter__(self) -> Generator[(Response, float), None, None]:
        for response_, probability_ in self._distribution.items():
            yield response_, probability_

    def __repr__(self) -> str:
        return self._distribution.__repr__()

    def choose_response(self) -> Response:
        responses = list(self._distribution.keys())
        probabilities = list(self._distribution.values())
        response_: Response = common.rng.choice(responses, p=probabilities)
        return response_


distribution: ResponseDistribution = ResponseDistribution()
distribution[response_1] = 0.4
distribution[response_2] = 0.6
print(distribution)
for response, probability_ in distribution:
    print(response, probability_)

print()
print(distribution.choose_response())
print(distribution.choose_response())
print(distribution.choose_response())
print(distribution.choose_response())
print(distribution.choose_response())
print(distribution.choose_response())


class Dynamics(dict[tuple[State, Action], ResponseDistribution]):
    """# p(s',r|s,a) function"""
    pass

    def add_transition(self, state: State, action: Action, new_state: State, reward: float, probability: float):
        response_distribution: ResponseDistribution = self.get((state, action), ResponseDistribution())
        if not response_distribution:
            self[(state, action)] = response_distribution
        response: Response = Response(new_state, reward)
        response_distribution[response] = probability


# my_dist: ResponseDistribution = ResponseDistribution()
# print(my_dist)
hello_state = State(name="hello")
left_action = Action(name="left")
right_action = Action(name="right")
goodbye_state = State(name="goodbye")
response_1 = Response(goodbye_state, 1.0)
my_dist[response_1] = 0.1
print(my_dist)
print(type(my_dist))

dynamics = Dynamics()



