from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model.environment.environment import Environment
from mdp.model.environment.state import State
from mdp.model.environment.action import Action
from mdp.model.environment.dynamics import Dynamics

S_A = tuple[int, int]


class EnvironmentContinuous(Environment):
    """An abstract Environment with continuous states but discrete actions"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        super().__init__(environment_parameters)

        # action list and action lookup
        self.actions: list[Action] = []
        self.action_index: dict[Action: int] = {}

    def build(self):
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self.dynamics.build()
        self._build_distributions()

    def _build_distributions(self):
        # TODO: decide what to do here, if anything, if start state is a continous distribution
        start_states: list[State] = self.dynamics.get_start_states()
        if len(start_states) == 1:
            self.start_state_distribution = common.SingularDistribution[State](start_states)
        else:
            self.start_state_distribution = common.UniformDistribution[State](start_states)

    # region Sets
    @abstractmethod
    def _build_actions(self):
        pass

    def _is_action_compatible_with_state(self, state: State, action: Action):
        return True
    # endregion

    # region Operation
    def insert_state_function_into_graph3d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction,
                                           parameter: Optional[any] = None):
        pass

    # def start_s(self) -> int:
    #     state = self.dynamics.get_a_start_state()
    #     return self.state_index[state]

    # def start_s(self) -> int:
    #     state = self._get_a_start_state()
    #     s: int = self.state_index[state]
    #     return s

    # def from_state_perform_action(self, state: State, action: Action) -> Response:
    #     if state.is_terminal:
    #         raise Exception("Environment: Trying to act in a terminal state.")
    #     if not self.is_action_compatible_with_state(state, action):
    #         raise Exception(f"_apply_action state {state} incompatible with action {action}")
    #     response: Response = self.dynamics.draw_response(state, action)
    #     return response

    def from_state_perform_action(self, state: State, action: Action) -> tuple[float, State]:
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self._is_action_compatible_with_state(state, action):
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        reward, new_state = self.dynamics.draw_response(state, action)
        if not new_state:
            new_state: State = self.start_state_distribution.draw_one()

        return reward, new_state

    def update_grid_value_functions(self,
                                    algorithm: Algorithm,
                                    policy: Policy):
        pass

    def is_valued_state(self, state: State) -> bool:
        return False
    # endregion
