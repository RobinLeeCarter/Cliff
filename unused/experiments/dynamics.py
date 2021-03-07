from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Generator

import random
# import numpy as np
# from mdp import common


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


class StateDistribution:
    """normalised probability distribution p(s') of possible next states from Environment for one (s,a)"""
    def __init__(self):
        self._states: list[State] = []          # s'
        self._probabilities: list[float] = []   # p(s'|s,a)
        self._reward_distributions: list[RewardDistribution] = []

        # duplication of lookup for performance
        self._state_lookup: dict[State, RewardDistribution] = {}

        self._expected_reward: float = 0.0      # E[r|s,a] = Sum over r,s' of r.p(s',r|s,a)
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

    def states(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
        for state, reward_distribution in zip(self._states, self._reward_distributions):
            yield state, reward_distribution.expected_reward, reward_distribution.state_probability

    def states_and_rewards(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple(s', r, p(s',r|s,a))"""
        for state, reward_distribution in zip(self._states, self._reward_distributions):
            for reward, probability in reward_distribution.rewards():
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
        self._state: Optional[State] = None
        self._action: Optional[Action] = None
        self._state_distribution: Optional[StateDistribution] = None

    @property
    def expected_reward(self) -> Optional[float]:
        if self._state_distribution:
            return self._state_distribution.expected_reward
        else:
            return None

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

    def set_state_action(self, state: State, action: Action):
        """set state and action before using the functions below"""
        self._state: State = state
        self._action: Action = action
        state_action: tuple[State, Action] = (state, action)
        self._state_distribution = self._state_distributions.get(state_action)

    def draw(self) -> (State, float):
        """get (s',r)"""
        return self._state_distribution.draw()

    def states(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
        yield from self._state_distribution.states()

    def states_and_rewards(self) -> Generator[(State, float, float), None, None]:
        """iterator for tuple(s', r, p(s',r|s,a))"""
        yield from self._state_distribution.states_and_rewards()

    # def draw(self, state: State, action: Action) -> (State, float):
    #     """pass (s,a) get (s',r)"""
    #     state_action: tuple[State, Action] = (state, action)
    #     state_distribution = self._state_distributions[state_action]
    #     return state_distribution.draw()

    # def get_expected_reward(self, state: State, action: Action):
    #     state_action: tuple[State, Action] = (state, action)
    #     state_distribution = self._state_distributions.get(state_action)
    #     return state_distribution.expected_reward

    # def states(self, state: State, action: Action) -> Generator[(State, float, float), None, None]:
    #     """pass (s,a) iterator for tuple(s', E[r|s,a,s'], p(s'|s,a))"""
    #     state_action: tuple[State, Action] = (state, action)
    #     state_distribution = self._state_distributions.get(state_action)
    #     yield from state_distribution.states()

    # def states_and_rewards(self, state: State, action: Action) -> Generator[(State, float, float), None, None]:
    #     """pass (s,a) iterator for tuple(s', r, p(s',r|s,a))"""
    #     state_action: tuple[State, Action] = (state, action)
    #     state_distribution = self._state_distributions.get(state_action)
    #     yield from state_distribution.states_and_rewards()


# RewardDistribution test
print("RewardDistribution")

r_dist = RewardDistribution()
r_dist.add(reward=10.0, probability=0.1)
r_dist.add(reward=20.0, probability=0.2)
r_dist.add(reward=30.0, probability=0.3)
r_dist.add(reward=40.0, probability=0.4)

print("reward values...")
for reward_, probability_ in r_dist.rewards():
    print(reward_, probability_)

print("drawing...")
print(r_dist.draw())
print(r_dist.draw())
print(r_dist.draw())
print(r_dist.draw())


# StateDistribution
print("StateDistribution")
hello_state = State(name="hello")
goodbye_state = State(name="goodbye")
left_action = Action(name="left")
right_action = Action(name="right")

s_dist = StateDistribution()
s_dist.add(hello_state, 10.0, 0.1)
s_dist.add(hello_state, 20.0, 0.2)
s_dist.add(goodbye_state, -30.0, 0.3)
s_dist.add(goodbye_state, -40.0, 0.4)

print("(new_state, reward) values...")
for new_state_, reward_, probability_ in s_dist.states_and_rewards():
    print(new_state_, reward_, probability_)

print("new_state values...")
for new_state_, expected_reward_, probability_ in s_dist.states():
    print(new_state_, expected_reward_, probability_)

print("state distribution values...")
print(f"expected_reward={s_dist.expected_reward}")
# noinspection PyProtectedMember
print(f"total_probability={s_dist._total_probability}")

print("drawing...")
print(s_dist.draw())
print(s_dist.draw())
print(s_dist.draw())
print(s_dist.draw())
print(s_dist.draw())

# Dynamics
print("Dynamics")
dynamics = Dynamics()
dynamics.add(state=hello_state, action=left_action, new_state=hello_state, reward=10.0, probability=0.25)
dynamics.add(state=hello_state, action=left_action, new_state=hello_state, reward=12.0, probability=0.25)
dynamics.add(state=hello_state, action=left_action, new_state=goodbye_state, reward=-11.0, probability=0.5)
dynamics.add(state=hello_state, action=right_action, new_state=goodbye_state, reward=20.0, probability=1.0)
dynamics.add(state=goodbye_state, action=left_action, new_state=hello_state, reward=30.0, probability=1.0)
dynamics.add(state=goodbye_state, action=right_action, new_state=goodbye_state, reward=40.0, probability=1.0)


def output_state_action(state: State, action: Action):
    print()
    print(f"({state.name}, {action.name}) values...")
    dynamics.set_state_action(state, action)

    print(f"expected_reward={dynamics.expected_reward}")

    print("new_state values...")
    for new_state, expected_reward, probability in dynamics.states():
        print(new_state, expected_reward, probability)

    print("new_state, reward values...")
    for new_state, reward, probability in dynamics.states_and_rewards():
        print(new_state, reward, probability)


output_state_action(hello_state, left_action)
output_state_action(hello_state, right_action)
output_state_action(goodbye_state, left_action)
output_state_action(goodbye_state, right_action)
