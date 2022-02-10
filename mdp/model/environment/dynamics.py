from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment
from mdp.model.environment.state import State

Response = tuple[float, State]


class Dynamics(ABC):
    def __init__(self, environment: Environment, environment_parameters: common.EnvironmentParameters):
        """init top down"""
        self._environment: Environment = environment
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self._verbose: bool = environment_parameters.verbose
        self.is_built: bool = False

    def build(self):
        """build bottom up"""
        self.is_built = True

    def get_start_state_distribution(self) -> common.Distribution[State]:
        """
        Starting state distribution
        If want to use something different to a Uniform list of States, override this method to return the distribution
        """
        start_states: list[State] = self.get_start_states()
        if start_states:
            if len(start_states) == 1:
                return common.SingularDistribution[State](start_states)
            else:
                return common.UniformMultinoulli[State](start_states)
        else:
            raise Exception("Empty list of start states so nowhere to start!")

    def get_start_states(self) -> list[State]:
        """If as simple as a list of start states then return it here and the distribution will be generated"""
        pass

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        """
        pass

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
