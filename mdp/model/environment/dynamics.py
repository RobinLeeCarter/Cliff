from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import numpy as np

if TYPE_CHECKING:
    # from mdp.model.policy.policy import Policy
    from mdp import common
    from mdp.common import Distribution
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.response import Response
    from mdp.model.environment.environment import Environment


class Dynamics(ABC):
    def __init__(self, environment: Environment, environment_parameters: common.EnvironmentParameters):
        """init top down"""
        self._environment: Environment = environment
        self._environment_parameters: common.EnvironmentParameters = environment_parameters
        self._verbose: bool = environment_parameters.verbose
        self.is_built: bool = False
        # state_transition_probabilities[s',s,a] = p(s'|s,a)
        self.state_transition_probabilities: np.ndarray = np.array([], dtype=np.float)
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        self.expected_reward_np: np.ndarray = np.array([], dtype=np.float)

    def build(self):
        """build bottom up"""
        self.is_built = True

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

    def get_state_transition_distribution(self, state: State, action: Action) -> Distribution[State]:
        """
        dict[ s', p(s'|s,a) ]
        distribution of next states for a (state, action)
        """
        pass

    # def get_state_transition_probability_matrix(self, policy: Policy) -> np.ndarray:
    #     pass

    def get_summary_outcomes(self, state: State, action: Action) -> Distribution[Response]:
        """
        dict of possible responses for a single state and action
        with the expected_reward given in place of reward
        """
        pass

    def get_all_outcomes(self, state: State, action: Action) -> Distribution[Response]:
        """
        dict of possible responses for a single state and action
        could be used for one state, action in theory
        but too many for all states and actions so potentially not useful in practice
        """
        pass

    @abstractmethod
    def get_a_start_state(self) -> State:
        pass

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        pass
