from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Optional

if TYPE_CHECKING:
    from mdp import common
from mdp.model import environment
from mdp.scenarios.racetrack import constants
from mdp.scenarios.racetrack.model import action, state, grid_world


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_array = constants.TRACK
        grid_world_ = grid_world.GridWorld(grid_array)
        super().__init__(environment_parameters, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[state.State] = self.states
        self.actions: list[action.Action] = self.actions
        self._state: state.State = self._state
        self._action: action.Action = self._action
        self.grid_world_: grid_world.GridWorld = self.grid_world_

        self._reward: float = 0.0

        # velocity
        self.min_vx: int = 0
        self.max_vx: int = constants.MAX_VELOCITY
        self.min_vy: int = 0
        self.max_vy: int = constants.MAX_VELOCITY

        # acceleration
        self.min_ax: int = constants.MIN_ACCELERATION
        self.max_ax: int = constants.MAX_ACCELERATION
        self.min_ay: int = constants.MIN_ACCELERATION
        self.max_ay: int = constants.MAX_ACCELERATION

    # region Sets
    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position: common.XY = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                for vx in range(self.min_vx, self.max_vx+1):
                    for vy in range(self.min_vy, self.max_vy+1):
                        new_state: state.State = state.State(
                            is_terminal=is_terminal,
                            position=position,
                            velocity=common.XY(x=vx, y=vy)
                        )
                        self.states.append(new_state)

    def _build_actions(self):
        for ax in range(self.min_ax, self.max_ax + 1):
            for ay in range(self.min_ay, self.max_ay + 1):
                new_action: action.Action = action.Action(
                    acceleration=common.XY(x=ax, y=ay)
                )
                self.actions.append(new_action)

    # possible need to materialise this if it's slow since it will be at the bottom of the loop
    def actions_for_state(self, state_: state.State) -> \
            Generator[action.Action, None, None]:
        """set A(s)"""
        for action_ in self.actions:
            if self.is_action_compatible_with_state(state_, action_):
                yield action_

    def is_action_compatible_with_state(self, state_: state.State, action_: action.Action):
        new_vx = state_.velocity.x + action_.acceleration.x
        new_vy = state_.velocity.y + action_.acceleration.y
        if self.min_vx <= new_vx <= self.max_vx and \
            self.min_vy <= new_vy <= self.max_vy and \
                not (new_vx == 0 and new_vy == 0):
            return True
        else:
            return False
    # endregion

    # region Operation
    def _get_a_start_state(self) -> state.State:
        position: common.XY = self.grid_world.get_a_start_position()
        return state.State(is_terminal=False, position=position, velocity=common.XY(x=0, y=0))

    def _apply_action(self):
        if not self.is_action_compatible_with_state(self._state, self._action):
            raise Exception(f"apply_action_to_state state {self._state} incompatible with action {self._action}")

        # apply grid world rules (eg. edges, wind)
        acceleration: Optional[common.XY] = None
        if self._action:
            acceleration = self._action.acceleration
        new_position, new_velocity = self.grid_world_.change_request(
            position=self._state.position,
            velocity=self._state.velocity,
            acceleration=acceleration
            )

        self._square = self.grid_world.get_square(new_position)
        if self._square == common.Square.END:
            # success
            self._reward = 0.0
            self._projected_state = state.State(
                position=new_position,
                velocity=new_velocity,
                is_terminal=True
            )
            if self.verbose:
                print(f"Past finish line at {new_position}")
        elif self._square == common.Square.GRASS:
            # failure, move back to start line
            # self.pre_reset_state = state.State(x, y, vx, vy, is_reset=True)
            self._reward = -1.0 + constants.EXTRA_REWARD_FOR_FAILURE
            self._projected_state = self._get_a_start_state()
            if self.verbose:
                print(f"Grass at {new_position}")
        else:
            # TRACK or START so continue
            self._reward = -1.0
            self._projected_state = state.State(
                position=new_position,
                velocity=new_velocity,
                is_terminal=False
            )

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=self._reward,
            state=self._projected_state
        )
    # endregion
