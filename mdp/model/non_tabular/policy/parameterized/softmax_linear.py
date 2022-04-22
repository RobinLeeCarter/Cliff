from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional

import numpy as np
from scipy import special

import utils
if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
from mdp import common
from mdp.model.non_tabular.policy.parameterized.vector_parameterized import VectorParameterized
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class SoftmaxLinear(VectorParameterized[State, Action],
                    policy_type=common.PolicyType.NON_TABULAR_SOFTMAX_LINEAR,
                    policy_name="softmax linear",
                    has_feature_vector=True):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 ):
        super().__init__(environment, policy_parameters)
        self._tau: float = policy_parameters.tau
        self._uses_tau: bool = (self._tau != 1.0)

        self._possible_actions: list[Action] = []
        self._feature_matrix: Optional[np.ndarray] = None

    def _draw_action(self, state: State) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        probabilities: np.ndarray = self.get_action_probabilities(state)
        index: int = utils.p_choice(probabilities)
        self._feature_vector = self._feature_matrix[index]
        return self._possible_actions[index]

    def get_probability(self, state: State, action: Action) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        probabilities: np.ndarray = self.get_action_probabilities(state)
        index: int = self._environment.action_index[action]
        self._feature_vector = self._feature_matrix[index]
        return probabilities[index]

    def get_action_probabilities(self, state: State) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        self._environment.build_possible_actions(state, build_array=False)
        self._possible_actions: list[Action] = self._environment.possible_actions_list
        preferences_array: np.ndarray = self.get_action_values(state, self._possible_actions)
        if self._uses_tau:
            # https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning
            preferences_array /= self._tau
        probabilities: np.ndarray = special.softmax(preferences_array)
        return probabilities

    def get_action_values(self, state: State, actions: list[Action]) -> np.ndarray:
        self._feature_matrix: np.ndarray = self._feature.get_matrix(state, actions)
        values_array: np.ndarray = self._feature.matrix_product(self._feature_matrix, self._theta)
        return values_array

    # def get_action_values3(self, state: State, actions: list[Action]) -> np.ndarray:
    #     values_array: np.ndarray = self._feature.get_dot_products(state, actions, self._theta)
    #     return values_array

