from __future__ import annotations
from typing import Generator, Optional, TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model import algorithm, policy
    from mdp.model.environment import grid_world
from mdp import common
from mdp.model.environment import state, response
from mdp.model.environment import action


class Environment(abc.ABC):
    """A GridWorld Environment - too hard to make general at this point"""
    def __init__(self,
                 environment_parameters: common.EnvironmentParameters,
                 grid_world_: grid_world.GridWorld):
        self._environment_parameters = environment_parameters
        self.grid_world: grid_world.GridWorld = grid_world_
        self.verbose: bool = environment_parameters.verbose

        # state and states
        self.states: list[state.State] = []
        self.state_index: dict[state.State: int] = {}
        self.state_type: type = state.State     # required?

        # action and actions
        self.actions: list[action.Action] = []
        self.action_index: dict[action.Action: int] = {}
        self.action_type: type = action.Action  # required?

        # for processing response
        self._state: Optional[state.State] = None
        self._action: Optional[action.Action] = None
        self._projected_position: Optional[common.XY] = None
        self._square: Optional[common.Square] = None
        self._projected_state: Optional[state.State] = None

        self._build()

    def _build(self):
        self._build_states()
        self.state_index = {state_: i for i, state_ in enumerate(self.states)}
        self._build_actions()
        self.action_index = {action_: i for i, action_ in enumerate(self.actions)}

    def state_action_index(self, state_: state.State, action_: action.Action) -> tuple[int, int]:
        state_index = self.state_index[state_]
        action_index = self.action_index[action_]
        return state_index, action_index

    # region Sets
    @abc.abstractmethod
    def _build_states(self):
        pass

    @abc.abstractmethod
    def _build_actions(self):
        pass

    # possible need to materialise this if it's slow since it will be at the bottom of the loop
    # noinspection PyUnusedLocal
    def actions_for_state(self, state_: state.State) -> Generator[action.Action, None, None]:
        """set A(s)"""
        for action_ in self.actions:
            # if self.is_action_compatible_with_state(state_, action_):
            yield action_

    # def get_action_from_index(self, index: tuple[int]) -> action.Action:
    #     return self._actions_object.get_action_from_index(index)

    # def is_action_compatible_with_state(self, state_: state.State, action_: action.Action):
    #     new_vx = state_.vx + action_.ax
    #     new_vy = state_.vy + action_.ay
    #     if self.min_vx <= new_vx <= self.max_vx and \
    #         self.min_vy <= new_vy <= self.max_vy and \
    #             not (new_vx == 0 and new_vy == 0):
    #         return True
    #     else:
    #         return False
    # endregion

    # region Operation
    def start(self) -> response.Response:
        state_ = self._get_a_start_state()
        # if self.verbose:
        #     self.trace_.start(state_)
        return response.Response(state=state_, reward=None)

    @abc.abstractmethod
    def _get_a_start_state(self) -> state.State:
        pass

    def from_state_perform_action(self, state_: state.State, action_: action.Action) -> response.Response:
        if state_.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        self._state = state_
        self._action = action_
        self._apply_action()
        return self._get_response()

    @abc.abstractmethod
    def _apply_action(self):
        pass

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

    @abc.abstractmethod
    def _get_response(self) -> response.Response:
        pass

    def update_grid_value_functions(self, algorithm_: algorithm.Episodic, policy_: policy.Policy):
        pass

    def is_valued_state(self, state_: state.State) -> bool:
        return False
    # endregion
