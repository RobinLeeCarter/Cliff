from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm_ import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function
from mdp import common

from mdp.model.environment.state import State
from mdp.model.environment.action import Action
from mdp.model.environment.response import Response
from mdp.model.environment.dynamics import Dynamics
from mdp.model.environment.grid_world import GridWorld


class Environment(ABC):
    """A GridWorld Environment - too hard to make general at this point"""
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        self._environment_parameters = environment_parameters
        self.verbose: bool = environment_parameters.verbose

        # state and states
        self.states: list[State] = []
        self.state_index: dict[State: int] = {}
        self.state_type: type = State     # required?

        # action and actions
        self.actions: list[Action] = []
        self.action_index: dict[Action: int] = {}
        self.action_type: type = Action  # required?
        self.actions_for_state: dict[State, list[Action]] = {}
        self.s_a_compatibility: np.ndarray = np.empty(0, dtype=bool)

        # for processing response
        # self._state: Optional[State] = None
        # self._action: Optional[Action] = None
        # self._reward: Optional[float] = None
        # self._new_state: Optional[State] = None
        # self._response: Optional[Response] = None

        self._square: Optional[common.Square] = None

        # None to ensure not used when not used/initialised
        self.dynamics: Optional[Dynamics] = None
        self.grid_world: Optional[GridWorld] = None

    def build(self):
        self._build_states()
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self._build_state_actions()
        self.dynamics.build()

    def state_action_index(self, state: State, action: Action) -> tuple[int, int]:
        state_index = self.state_index[state]
        action_index = self.action_index[action]
        return state_index, action_index

    # region Sets
    @abstractmethod
    def _build_states(self):
        pass

    @abstractmethod
    def _build_actions(self):
        pass

    def _build_state_actions(self):
        """materialise A(s)"""
        self.s_a_compatibility = np.empty(shape=(len(self.states), len(self.actions),), dtype=bool)
        for s, state in enumerate(self.states):
            actions_for_state: list[Action] = []
            for a, action in enumerate(self.actions):
                compatible: bool = self.is_action_compatible_with_state(state, action)
                self.s_a_compatibility[s, a] = compatible
                if compatible:
                    actions_for_state.append(action)
            self.actions_for_state[state] = actions_for_state

    def is_action_compatible_with_state(self, state: State, action: Action):
        return True
    # endregion

    # region Operation
    def initialize_policy(self, policy_: Policy, policy_parameters: common.PolicyParameters):
        pass

    def insert_state_function_into_graph3d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction,
                                           parameter: Optional[any] = None):
        pass

    def start(self) -> Response:
        state = self._get_a_start_state()
        return Response(state=state, reward=None)

    def start_s(self) -> int:
        state = self._get_a_start_state()
        s: int = self.state_index[state]
        return s

    def _get_a_start_state(self) -> State:
        return self.dynamics.get_a_start_state()

    def from_state_perform_action(self, state: State, action: Action) -> Response:
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self.is_action_compatible_with_state(state, action):
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        response: Response = self.dynamics.draw_response(state, action)
        return response

    def from_s_perform_a(self, s: int, a: int) -> tuple[float, int]:
        state = self.states[s]
        action = self.actions[a]
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if not self.s_a_compatibility[s, a]:
            raise Exception(f"_apply_action state {state} incompatible with action {action}")
        response = self.dynamics.draw_response(state, action)
        new_s = self.state_index[self]
        return response.reward, new_s

    # TODO: move down hierarchy, too general for top level
    def _project_back_to_grid(self, requested_position: common.XY) -> common.XY:
        x = requested_position.x
        y = requested_position.y
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.grid_world.max_x:
            x = self.grid_world.max_x
        if y > self.grid_world.max_y:
            y = self.grid_world.max_y
        return common.XY(x=x, y=y)

    def update_grid_value_functions(self,
                                    algorithm_: Algorithm,
                                    policy_: Policy):
        pass

    def is_valued_state(self, state: State) -> bool:
        return False

    def output_mode(self):
        pass
    # endregion
