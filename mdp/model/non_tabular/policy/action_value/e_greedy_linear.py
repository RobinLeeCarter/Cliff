from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

import utils
if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction
from mdp.model.non_tabular.policy.action_value.action_value_policy import ActionValuePolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class EGreedyLinear(ActionValuePolicy[State, Action]):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 state_action_function: StateActionFunction[State, Action],
                 epsilon: float = 0.0
                 ):
        super().__init__(environment, policy_parameters, state_action_function)
        self.epsilon: float = epsilon
        self.is_deterministic: bool = (epsilon == 0.0)

    def _draw_action(self, state: State) -> Action:
        """"
        :param state: starting state
        :return: action drawn from probability distribution pi(state, action; theta)
        """
        self._environment.build_possible_actions(state, build_array=False)
        possible_actions: list[Action] = self._environment.possible_actions_list
        action_values: np.ndarray = self._state_action_function.get_action_values(state, possible_actions)
        # could also jit this if needed
        if utils.uniform() > self.epsilon:
            max_index: int = int(np.argmax(action_values))
            return possible_actions[max_index]
        else:
            i = utils.n_choice(len(possible_actions))
            return possible_actions[i]

    def get_probability(self, state: State, action: Action) -> float:
        """
        :param state: State
        :param action: Action
        :return: probability of taking action from state
        """
        action_index: int = self._environment.action_index[action]

        self._environment.build_possible_actions(state, build_array=False)
        possible_action_list: list[Action] = self._environment.possible_actions_list

        action_values: np.ndarray = self._state_action_function.get_action_values(state, possible_action_list)
        max_index: int = int(np.argmax(action_values))

        non_greedy_p = self.epsilon / len(possible_action_list)
        if max_index == action_index:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p

    def get_action_probabilities(self, state: State) -> np.ndarray:
        """
        :param state: State
        :return: probability distribution of all actions across list of standard actions for environment
        """
        self._probabilities.fill(0.0)

        build_array: bool = not self.is_deterministic   # if determinstic then don't need array
        self._environment.build_possible_actions(state, build_array)
        possible_action_list: list[Action] = self._environment.possible_actions_list

        action_values: np.ndarray = self._state_action_function.get_action_values(state, possible_action_list)
        max_index: int = int(np.argmax(action_values))

        if self.is_deterministic:
            self._probabilities[max_index] = 1.0
            return self._probabilities
        else:
            non_greedy_p = self.epsilon / len(possible_action_list)
            greedy_p = (1 - self.epsilon) + non_greedy_p
            possible_actions_array: np.ndarray = self._environment.possible_actions_array
            self._probabilities[possible_actions_array] = non_greedy_p
            self._probabilities[max_index] = greedy_p
            return self._probabilities
