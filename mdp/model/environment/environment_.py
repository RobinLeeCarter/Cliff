from __future__ import annotations
from typing import Generator, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model import algorithm, policy
    from mdp.model.algorithm.value_function import state_function
from mdp import common

from mdp.model.environment.state import State
from mdp.model.environment.action import Action
from mdp.model.environment.response import Response
from mdp.model.environment.dynamics import Dynamics
from mdp.model.environment.grid_world import GridWorld


class Environment(ABC):
    """A GridWorld Environment - too hard to make general at this point"""
    def __init__(self,
                 environment_parameters: common.EnvironmentParameters,
                 grid_world: Optional[GridWorld] = None):
        self._environment_parameters = environment_parameters
        self.grid_world: Optional[GridWorld] = grid_world
        self.verbose: bool = environment_parameters.verbose

        # state and states
        self.states: list[State] = []
        self.state_index: dict[State: int] = {}
        self.state_type: type = State     # required?

        # action and actions
        self.actions: list[Action] = []
        self.action_index: dict[Action: int] = {}
        self.action_type: type = Action  # required?

        # for processing response
        self._state: Optional[State] = None
        self._action: Optional[Action] = None
        self._reward: Optional[float] = None
        self._new_state: Optional[State] = None
        self._response: Optional[Response] = None

        self._square: Optional[common.Square] = None

        # None to ensure not used when not used/initialised
        self._dynamics: Optional[Dynamics] = None

    def build(self):
        self._build_states()
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self._build_actions()
        self.action_index = {action: i for i, action in enumerate(self.actions)}
        self._build_dynamics()

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

    # possible need to materialise this if it's slow since it will be at the bottom of the loop
    def actions_for_state(self, state: State) -> Generator[Action, None, None]:
        """set A(s)"""
        for action in self.actions:
            if self.is_action_compatible_with_state(state, action):
                yield action

    def is_action_compatible_with_state(self, state: State, action: Action):
        return True

    def _build_dynamics(self):
        if self._dynamics:
            self._dynamics.build()
    # endregion

    # region Operation
    def initialize_policy(self, policy_: policy.Policy, policy_parameters: common.PolicyParameters):
        pass

    def insert_state_function_into_graph3d(self, comparison: common.Comparison, v: state_function.StateFunction):
        pass

    def start(self) -> Response:
        state = self._get_a_start_state()
        # if self.verbose:
        #     self.trace_.start(state)
        return Response(state=state, reward=None)

    @abstractmethod
    def _get_a_start_state(self) -> State:
        pass

    def from_state_perform_action(self, state: State, action: Action) -> Response:
        if state.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        self._state = state
        self._action = action
        self._apply_action()
        return self._get_response()

    def _apply_action(self):
        self._response = self._dynamics.draw_response(self._state, self._action)

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

    def _get_response(self) -> Response:
        return self._response

    def update_grid_value_functions(self, algorithm_: algorithm.Algorithm, policy_: policy.Policy):
        pass

    def is_valued_state(self, state: State) -> bool:
        return False

    def output_mode(self):
        pass
    # endregion
