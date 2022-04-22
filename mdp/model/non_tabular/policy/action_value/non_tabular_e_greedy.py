from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeVar

import numpy as np

import utils


if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
from mdp import common
from mdp.model.non_tabular.policy.action_value.action_value_policy import ActionValuePolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.value_function.state_action.linear_state_action_function import LinearStateActionFunction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class NonTabularEGreedy(ActionValuePolicy[State, Action],
                        policy_type=common.PolicyType.NON_TABULAR_E_GREEDY,
                        policy_name="ε-greedy"):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 ):
        super().__init__(environment, policy_parameters)
        self.epsilon: float = self._policy_parameters.epsilon
        self.is_deterministic: bool = (self.epsilon == 0.0)
        # if environment.actions_always_compatible:

    def _draw_action(self, state: State) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        self._environment.build_possible_actions(state, build_array=False)
        possible_actions: list[Action] = self._environment.possible_actions_list
        action_values: np.ndarray = self._Q.get_action_values(state, possible_actions)
        # could also jit this if needed
        if utils.unit_uniform() > self.epsilon:
            index: int = utils.choose_argmax_index(action_values)
        else:
            index: int = utils.n_choice(len(possible_actions))

        if self._Q.has_feature_matrix:
            assert isinstance(self._Q, LinearStateActionFunction)
            self._feature_vector = self._Q.feature_matrix[index]

        return possible_actions[index]

    def get_probability(self, state: State, action: Action) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        action_index: int = self._environment.action_index[action]

        self._environment.build_possible_actions(state, build_array=False)
        possible_action_list: list[Action] = self._environment.possible_actions_list
        non_greedy_p = self.epsilon / len(possible_action_list)

        action_values: np.ndarray = self._Q.get_action_values(state, possible_action_list)
        if self._Q.has_feature_matrix:
            assert isinstance(self._Q, LinearStateActionFunction)
            self._feature_vector = self._Q.feature_matrix[action_index]
        max_indices: np.ndarray = np.flatnonzero(action_values == np.max(action_values))

        if action_index in max_indices:
            greedy_p = ((1 - self.epsilon) / max_indices.size) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p

    def get_action_probabilities(self, state: State) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        print("get_action_probabilities")
        self._probabilities.fill(0.0)

        build_array: bool = not self.is_deterministic   # if determinstic then don't need array
        self._environment.build_possible_actions(state, build_array)
        possible_action_list: list[Action] = self._environment.possible_actions_list

        action_values: np.ndarray = self._Q.get_action_values(state, possible_action_list)
        max_indices: np.ndarray = np.flatnonzero(action_values == np.max(action_values))

        if self.is_deterministic:
            self._probabilities[max_indices] = 1.0 / max_indices.size
            return self._probabilities
        else:
            non_greedy_p = self.epsilon / len(possible_action_list)
            greedy_p = ((1 - self.epsilon) / max_indices.size) + non_greedy_p
            possible_actions_array: np.ndarray = self._environment.possible_actions_array
            self._probabilities[possible_actions_array] = non_greedy_p
            self._probabilities[max_indices] = greedy_p
            assert math.isclose(self._probabilities.sum(), 1.0)
            return self._probabilities
