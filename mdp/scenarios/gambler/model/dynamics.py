from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenarios.gambler.model.action import Action
    from mdp.scenarios.gambler.model.environment import Environment
    from mdp.scenarios.gambler.model.environment_parameters import EnvironmentParameters

import numpy as np

from mdp.common import Distribution, UniformDistribution
from mdp.model.environment import dynamics

from mdp.scenarios.gambler.model.state import State
from mdp.scenarios.gambler.model.enums import Toss, Result


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._environment_parameters: EnvironmentParameters = self._environment_parameters
        self._probability_heads: float = self._environment_parameters.probability_heads
        self._toss_distribution: Distribution[Toss] = Distribution()
        self._start_distribution: Optional[UniformDistribution[State]] = None

    def build(self):
        self._toss_distribution[Toss.HEADS] = self._probability_heads
        self._toss_distribution[Toss.TAILS] = 1.0 - self._probability_heads
        self._toss_distribution.enable()

        non_terminal_states = [state for state in self._environment.states if not state.is_terminal]
        self._start_distribution = UniformDistribution[State](non_terminal_states)

        self._build_state_transition_probabilities()
        self._build_expected_reward()

        super().build()

    def _build_state_transition_probabilities(self):
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        # (state, action, next_state)
        tensor_shape = (state_count, action_count, state_count)
        self.state_transition_probabilities = np.zeros(shape=tensor_shape, dtype=np.float)
        for s0, state in enumerate(self._environment.states):
            for action in self._environment.actions_for_state[state]:
                a0 = self._environment.action_index[action]
                next_state_distribution = self.get_state_transition_distribution(state, action)
                for next_state, probability in next_state_distribution.items():
                    s1 = self._environment.state_index[next_state]
                    self.state_transition_probabilities[s0, a0, s1] = probability

    def _build_expected_reward(self):
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        # (state, action)
        self.expected_reward_np = np.zeros(shape=(state_count, action_count), dtype=np.float)
        for s, state in enumerate(self._environment.states):
            for action in self._environment.actions_for_state[state]:
                a = self._environment.action_index[action]
                self.expected_reward_np[s, a] = self.get_expected_reward(state, action)

    def get_a_start_state(self) -> State:
        return self._start_distribution.draw_one()
        # return random.choice([state for state in self._environment.states if not state.is_terminal])

    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        r(s,a) = E[Rt | S(t-1)=s, A(t-1)=a] = Sum_over_s'_r( p(s',r|s,a).r )
        expected reward for a (state, action)
        """
        if state.capital + action.stake == 100:
            expected_reward = self._probability_heads   # (* 1.0)
        else:
            expected_reward = 0.0
        return expected_reward

    def get_state_transition_distribution(self, state: State, action: Action) -> Distribution[State]:
        """
        dict[ s', p(s'|s,a) ]
        distribution of next states for a (state, action)
        """
        distribution: Distribution[State] = Distribution()
        for toss in [Toss.HEADS, Toss.TAILS]:
            probability = self._toss_distribution[toss]
            if toss == Toss.HEADS:
                new_capital = state.capital + action.stake
            else:
                new_capital = state.capital - action.stake
            is_terminal: bool = (new_capital == 0 or new_capital == self._environment_parameters.max_capital)
            next_state = State(is_terminal=is_terminal, capital=new_capital)
            distribution[next_state] = probability
        distribution.enable(do_self_check=False)

        return distribution

    def draw_response(self, state: State, action: Action) -> tuple[float, State]:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        toss: Toss = self._toss_distribution.draw_one()
        if toss == Toss.HEADS:
            new_capital = state.capital + action.stake
        else:
            new_capital = state.capital - action.stake

        result: Optional[Result] = None
        reward: float = 0.0

        if new_capital == self._environment_parameters.max_capital:
            result = Result.WIN
            reward = 1.0
        elif new_capital == 0:
            result = Result.LOSE

        if self._verbose:
            print(f"starting_capital = {state.capital}")
            print(f"stake = {action.stake}")
            print(f"toss = {toss}")
            print(f"new_capital = {new_capital}")
            print(f"result = {result}")

        is_terminal: bool = (result is not None)
        new_state = State(is_terminal=is_terminal, capital=new_capital)
        return reward, new_state
