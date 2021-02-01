from typing import Generator
import numpy as np

import common
from environment import action, observation, state, grid


class Environment:
    def __init__(self, grid_: grid.Grid, rng: np.random.Generator, verbose: bool = False):
        self.rng: np.random.Generator = rng
        self.verbose: bool = verbose
        self.grid_world: grid.GridWorld = grid.GridWorld(grid_, rng)

        # position
        self.min_x: int = 0
        self.max_x: int = self.grid_world.grid.max_x
        self.min_y: int = 0
        self.max_y: int = self.grid_world.grid.max_y

        self.states_shape: tuple = (self.max_x + 1, self.max_y + 1)
        # self.action_list: List[action.Action] = [action_ for action_ in self.actions()]
        # self.action_dict: Dict[action.Action, int] = {action_: i for i, action_ in enumerate(self.actions())}
        # actions = [action_ for action_ in self.actions()]
        self.actions_shape: tuple = action.Actions.shape
        # self.trace_ = grid.Trace = grid.Trace(self.grid_world)

        # pre-reset state (if not None it means the state has just been reset and this was the failure state)
        # self.pre_reset_state: Optional[state.State] = None

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
        for action_ in action.Actions.action_list:
            yield action_

    # def get_index_from_action(self, action_: action.Action) -> tuple[int]:
    #     tuple_index = (self.action_dict[action_], )
    #     return tuple_index

    # def get_action_from_index(self, index: int) -> action.Action:
    #     action_ = self.action_list[index]
    #     return action_

    # possible need to materialise this if it's slow since it will be at the bottom of the loop
    # noinspection PyUnusedLocal
    def actions_for_state(self, state_: state.State) -> Generator[action.Action, None, None]:
        """set A(s)"""
        for action_ in self.actions():
            # if self.is_action_compatible_with_state(state_, action_):
            yield action_

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
        # if not self.is_action_compatible_with_state(state_, action_):
        #     raise Exception(f"from_state_perform_action state {state_} incompatible with action {action_}")

        wind: common.XY = self.grid_world.get_wind(state_.position)
        combined: common.XY = common.XY(
            x=action_.move.x + wind.x,
            y=action_.move.y + wind.y
        )
        blown_position: common.XY = common.XY(
            x=state_.position.x + combined.x,
            y=state_.position.y + combined.y
        )
        projected_position: common.XY = self._project_back_to_grid(blown_position)
        is_terminal: bool = self.grid_world.is_at_goal(projected_position)
        new_state: state.State = state.State(projected_position, is_terminal)
        reward: float = -1.0
        return observation.Observation(reward, new_state)

    def _project_back_to_grid(self, blown_position: common.XY) -> common.XY:
        x = blown_position.x
        y = blown_position.y
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.max_x:
            x = self.max_x
        if y > self.max_y:
            y = self.max_y
        return common.XY(x=x, y=y)
    # endregion
