from __future__ import annotations
from typing import Generator, TYPE_CHECKING

import common
from environment import action, observation, state, grid_world
if TYPE_CHECKING:
    from environment import grid


class Environment:
    """A GridWorld Environment - too hard to make general at this point"""
    def __init__(self, grid_: grid.Grid, verbose: bool = False):
        self.grid_world: grid_world.GridWorld = grid_world.GridWorld(grid_)
        self.verbose: bool = verbose

        self.states_shape: tuple = (self.grid_world.max_x + 1, self.grid_world.max_y + 1)
        self._actions: action.Actions = action.actions
        self.actions_shape: tuple = self._actions.shape

    # region Sets
    def states(self) -> Generator[state.State, None, None]:
        """set S"""
        for x in range(self.states_shape[0]):
            for y in range(self.states_shape[1]):
                position = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                yield state.State(position, is_terminal)

    def actions(self) -> Generator[action.Action, None, None]:
        """set A - same for all s in this scenario"""
        for action_ in self._actions.action_list:
            yield action_

    # possible need to materialise this if it's slow since it will be at the bottom of the loop
    # noinspection PyUnusedLocal
    def actions_for_state(self, state_: state.State) -> Generator[action.Action, None, None]:
        """set A(s)"""
        for action_ in self.actions():
            # if self.is_action_compatible_with_state(state_, action_):
            yield action_

    def get_action_from_index(self, index: tuple[int]) -> action.Action:
        return self._actions.get_action_from_index(index)

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
    def start(self) -> observation.Observation:
        state_ = self.get_a_start_state()
        # if self.verbose:
        #     self.trace_.start(state_)
        return observation.Observation(state=state_, reward=None)

    def get_a_start_state(self) -> state.State:
        position: common.XY = self.grid_world.get_a_start_position()
        return state.State(position)

    def from_state_perform_action(self, state_: state.State, action_: action.Action) -> observation.Observation:
        if state_.is_terminal:
            raise Exception("Environment: Trying to act in a terminal state.")
        if action_ is None:
            raise Exception("Environment: Action passed is none.")

        new_state: state.State

        # apply grid world rules (eg. edges, wind)
        projected_position: common.XY = self.grid_world.change_request(current=state_.position,
                                                                       request=action_.move)

        # tests
        square: common.Square = self.grid_world.get_square(projected_position)
        if square == common.Square.END:
            is_terminal = True
        else:
            is_terminal = False

        # reponse
        if square == common.Square.CLIFF:
            reward: float = -100.0
            new_state = self.get_a_start_state()
        else:
            reward: float = -1.0
            new_state: state.State = state.State(projected_position, is_terminal)
        return observation.Observation(reward, new_state)

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
    # endregion
