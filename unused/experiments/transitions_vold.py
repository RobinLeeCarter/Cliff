from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Generator

import random
import numpy as np

from mdp import common

rng: np.random.Generator = np.random.default_rng()

@dataclass(frozen=True)
class State:
    name: str


@dataclass(frozen=True)
class Action:
    name: str


@dataclass(frozen=True)
class Response:
    """(s',r) = (new_state, reward)"""
    state: state.NonTabularState
    reward: Optional[float]


# class _RewardProbDict(dict[float, float]):
#      def __missing__(self, key):
#          return 0.0

class RewardDistribution:
    """un-normalised probability distribution p(r) of possible rewards from Environment for one (s,a,s')"""
    def __init__(self):
        # dictionary for build
        # key on float is not ideal but also not a problem if 2 keys created for the same value
        # self._distribution: _RewardProbDict = _RewardProbDict()
        # self._distribution: dict[float, float] = {}

        # using a list rather than a dictionary as 20-30% faster for draw() which is called with high frequency
        self._rewards: list[float] = []
        self._probabilities: list[float] = []    # un-normalised probabilities

        self.expected_reward: float = 0.0
        self.total_probability: float = 0.0
        # self._normalised_probabilities: np.ndarray = np.array([], dtype=float)

    def add(self, reward: float, probability: float):
        # self._distribution[reward] += probability

        # probability = self.get((state, action), ResponseDistribution())
        # if not response_distribution:
        #     self[(state, action)] = response_distribution

        if reward in self._rewards:
            index = self._rewards.index(reward)
            self._probabilities[index] += probability
        else:
            self._rewards.append(reward)
            self._probabilities.append(probability)

        # if reward in self._distribution:
        #     self._distribution[reward] += probability
        #     index = self._rewards.index(reward)
        # else:
        #     self._distribution[reward] = probability


        self.expected_reward += reward * probability
        self.total_probability += probability
        # self._normalised_probabilities = np.array(self.probabilities, dtype=float) / self._total_probability

    def draw(self) -> float:
        # if not self._rewards:
        #     self._rewards = list(self._distribution.keys())
        #     self._probabilities = list(self._distribution.values())
        # common.rng.choice(self.rewards, p=self._normalised_probabilities)
        return random.choices(self._rewards, weights=self._probabilities)[0]


# class _StateRewardDistributionDict(dict[State, RewardDistribution]):
#      def __missing__(self, key):
#          return RewardDistribution()

class StateDistribution:
    """normalised probability distribution p(s') of possible next states from Environment for one (s,a)"""
    def __init__(self):
        # or dict[Response, float] but reward is float so unwise
        self._distribution: dict[State, RewardDistribution] = {}
        # self._distribution: dict[State, float] = {}

    def add_response_probability(self, response_: Response, probability: float):
        new_state = response_.state
        reward = response_.reward

        reward_distribution: RewardDistribution
        if new_state in self._distribution:
            reward_distribution = self._distribution[new_state]
        else:
            reward_distribution = RewardDistribution()
            self._distribution[new_state] = reward_distribution
        reward_distribution.add(reward, probability)

        # reward_distribution: RewardDistribution = self._distribution[response_.state]
        # reward_distribution.add(response_.reward, probability)

        # go faster code - not really needed
        # reward_distribution: RewardsDistribution = self.get((state, action), ResponseDistribution())
        # if not response_distribution:
        #     self[(state, action)] = response_distribution
        # reward_distribution: RewardDistribution

    def get_state_probability(self, new_state: State) -> float:
        return self._distribution[new_state].total_probability

    def get_state_expected_rewards(self, new_state: State) -> float:
        return self._distribution[new_state].expected_reward


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
        response_: Response = rng.choice(responses, p=probabilities)
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
# my_dist[response_1] = 0.1
# print(my_dist)
# print(type(my_dist))

dynamics = Dynamics()



