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
