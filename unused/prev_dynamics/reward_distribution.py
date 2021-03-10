from __future__ import annotations
# from typing import TYPE_CHECKING
from typing import Generator

import random


class RewardDistribution:
    """un-normalised probability distribution p(r) of possible rewards from Environment for one (s,a,s')"""
    def __init__(self):
        # using a list rather than a dictionary as 20-30% faster for draw() which is called with high frequency
        self._rewards: list[float] = []             # r
        self._probabilities: list[float] = []       # p(s',r|s,a)

        self.reward_times_p: float = 0.0  # Sum over r of r.p(s',r|s,a)
        self.state_probability: float = 0.0         # p(s'|s,a)

    @property
    def expected_reward(self) -> float:             # E[r|s,a,s'] = Sum over r of r.p(s',r|s,a) / p(s'|s,a)
        return self.reward_times_p / self.state_probability

    def add(self, reward: float, probability: float):
        if reward in self._rewards:
            index = self._rewards.index(reward)
            self._probabilities[index] += probability
        else:
            self._rewards.append(reward)
            self._probabilities.append(probability)

        self.reward_times_p += reward * probability
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

    def rewards(self) -> Generator[(float, float), None, None]:
        """r, p(s',r|s,a)"""
        for reward, probability in zip(self._rewards, self._probabilities):
            yield reward, probability

    def draw(self) -> float:
        """draw r from rewards distribution"""
        return random.choices(self._rewards, weights=self._probabilities)[0]
