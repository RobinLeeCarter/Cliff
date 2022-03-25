from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy import special

import utils
if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.feature.feature import Feature
from mdp.model.non_tabular.policy.parameterized.vector_parameterized import VectorParameterized
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class SoftmaxLinear(VectorParameterized[State, Action]):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 feature: Feature[State, Action],
                 initial_theta: float = 0.0,
                 tau: float = 1.0  # temperature parameter
                 ):
        super().__init__(environment, policy_parameters, feature, initial_theta)
        self._tau: float = tau
        self._uses_tau: bool = (tau != 1.0)

    def _draw_action(self, state: State) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        probabilities: np.ndarray = self.get_action_probabilities(state)
        choice: int = utils.p_choice(probabilities)
        return self._environment.actions[choice]

    def get_probability(self, state: State, action: Action) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        probabilities: np.ndarray = self.get_action_probabilities(state)
        index: int = self._environment.action_index[action]
        return probabilities[index]

    def get_action_probabilities(self, state: State) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        preferences: list[float] = []
        probabilities: np.ndarray
        self._feature.set_state(state)
        for action in self._environment.actions:
            self._feature.set_action(action)
            preference: float = self._feature.dot_product_full_vector(self._theta)
            preferences.append(preference)
        preferences_array = np.array(preferences)
        if self._uses_tau:
            # https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning
            preferences_array /= self._tau
        probabilities: np.ndarray = special.softmax(preferences_array)
        return probabilities
