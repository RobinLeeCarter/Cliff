from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC
import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.common import Multinoulli
    from mdp.model.environment.action import Action
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
from mdp.model.environment.dynamics import Dynamics
from mdp.model.environment.state import State

Response = tuple[float, State]


class TabularDynamics(Dynamics, ABC):
    def __init__(self, environment: TabularEnvironment, environment_parameters: common.EnvironmentParameters):
        """init top down"""
        super().__init__(environment, environment_parameters)
        self._environment: TabularEnvironment = environment

        # state_transition_probabilities[s',s,a] = p(s'|s,a)
        self.state_transition_probabilities: np.ndarray = np.array([], dtype=np.float)
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        self.expected_reward: np.ndarray = np.array([], dtype=np.float)

    def get_start_s_distribution(self) -> common.Distribution[int]:
        """
        Starting state distribution
        If want to use something different to a Uniform list of States override this method to return the distribution
        """
        start_states: list[State] = self.get_start_states()
        if start_states:
            start_s = [self._environment.state_index[state] for state in start_states]
            if len(start_states) == 1:
                return common.SingularDistribution[int](start_s)
            else:
                return common.UniformMultinoulli[int](start_s)
        else:
            raise Exception("Empty list of start states so nowhere to start!")

    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        r(s,a) = E[Rt | S(t-1)=s, A(t-1)=a] = Sum_over_s'_r( p(s',r|s,a).r )
        expected reward for a (state, action)
        """
        pass

    def get_expected_conditional_reward(self, state: State, action: Action, next_state: State) -> float:
        """
        r(s,a,s') = E[Rt | S(t)=s', S(t-1)=s, A(t-1=a)] = Sum_over_r( p(s',r|s,a).r ) / p(s'|s,a)
        expected reward for a (state, action) given the next state
        """
        probability_x_reward: float = self.get_probability_x_reward(state, action, next_state)
        next_state_probability: float = self.get_state_transition_probability(state, action, next_state)
        if next_state_probability == 0.0:
            return 0.0
        else:
            return probability_x_reward / next_state_probability

    def get_probability_x_reward(self, state: State, action: Action, next_state: State) -> float:
        """
        Sum_over_r( p(s',r|s,a).r )
        probability_x_reward for a (state, action) given the next state
        """
        pass

    def get_state_transition_probability(self, state: State, action: Action, next_state: State) -> float:
        """
        p(s'|s,a) = Sum_over_r( p(s',r|s,a) )
        probability of a next state for a (state, action)
        """
        pass

    def get_state_transition_distribution(self, state: State, action: Action) -> Multinoulli[State]:
        """
        dict[ s', p(s'|s,a) ]
        distribution of next states for a (state, action)
        """
        pass

    def get_summary_outcomes(self, state: State, action: Action) -> Multinoulli[Response]:
        """
        dict of possible responses for a single state and action
        with the expected_reward given in place of reward
        """
        pass

    def get_all_outcomes(self, state: State, action: Action) -> Multinoulli[Response]:
        """
        dict of possible responses for a single state and action
        could be used for one state, action in theory
        but too many for all states and actions so potentially not useful in practice
        """
        pass
