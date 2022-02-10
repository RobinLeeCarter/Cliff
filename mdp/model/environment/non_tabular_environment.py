from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model.environment.environment import Environment
from mdp.model.environment.state import State
from mdp.model.environment.action import Action

S_A = tuple[int, int]


class NonTabularEnvironment(Environment, ABC):
    """An abstract Environment with continuous states but discrete actions"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        """
        :param environment_parameters: the parameters specific to the specific environment e.g. number_of_cars
        """
        super().__init__(environment_parameters)

        # action list and action lookup
        self.actions: list[Action] = []
        self.action_index: dict[Action: int] = {}

        # Distributions
        self.start_state_distribution: Optional[common.Distribution[State]] = None

    def build(self):
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self.dynamics.build()
        self._build_distributions()

    def _build_distributions(self):
        self.start_state_distribution = self.dynamics.get_start_distribution()

    # region Sets
    @abstractmethod
    def _build_actions(self):
        pass
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
    # endregion
