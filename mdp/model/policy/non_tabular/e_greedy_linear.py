from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

import utils
if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.action import Action
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
    from mdp.model.feature.feature import Feature
    from mdp.model.algorithm.non_tabular.value_function.state_action_function import StateActionFunction
from mdp.model.policy.non_tabular.non_tabular_policy import NonTabularPolicy


class EGreedyLinear(NonTabularPolicy):
    # TODO: Write next
    def __init__(self,
                 environment: NonTabularEnvironment,
                 policy_parameters: common.PolicyParameters,
                 state_action_function: StateActionFunction,
                 epsilon: float = 0.0  # temperature parameter
                 ):
        super().__init__(environment, policy_parameters)
        self._state_action_function: StateActionFunction = state_action_function
        self._epsilon: float = epsilon
        self._uses_epsilon: bool = (epsilon != 0.0)

    def _draw_action(self, state: NonTabularState) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        possible_actions: list[NonTabularAction] = self._environment.get_possible_actions(state)
        action_values: np.ndarray = self._state_action_function.get_action_values(state, possible_actions)
        # could also jit this if needed
        if utils.uniform() > self._epsilon:
            max_index: int = int(np.argmax(action_values))
            return possible_actions[max_index]
        else:
            i = utils.n_choice(len(possible_actions))
            return possible_actions[i]

    def get_probability(self, state: NonTabularState, action: NonTabularAction) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        probabilities: np.ndarray = self.get_action_probabilities(state)
        index: int = self._environment.action_index[action]
        return probabilities[index]

    def get_action_probabilities(self, state: NonTabularState) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        # TODO: Finish this
        possible_actions: list[NonTabularAction] = self._environment.possible_actions(state)
        action_values: np.ndarray = self._state_action_function.get_action_values(state, possible_actions)


        non_greedy_p = self._epsilon * self._environment.one_over_possible_actions(state)
        # could also jit this if needed
        if utils.uniform() > self._epsilon:
            max_index: int = int(np.argmax(action_values))
            greedy_action: NonTabularAction = possible_actions[max_index]
            return greedy_action
        else:
            i = utils.n_choice(len(possible_actions))
            return possible_actions[i]

        preferences: list[float] = []
        probabilities: np.ndarray
        self._feature.state = state
        for action in self._environment.actions:
            self._feature.action = action
            preference: float = self._feature.dot_product_full_vector(self._theta)
            preferences.append(preference)
        preferences_array = np.array(preferences)
        if self._uses_tau:
            # https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning
            preferences_array /= self._tau
        probabilities: np.ndarray = special.softmax(preferences_array)
        return probabilities
