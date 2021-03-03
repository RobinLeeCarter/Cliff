from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model import environment
from mdp.scenarios.racetrack import grid_world, constants, state, action


class RacetrackEnvironment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_array = constants.TRACK
        grid_world_ = grid_world.RacetrackGridWorld(grid_array)
        super().__init__(environment_parameters, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[state.StatePositionVelocity] = self.states
        self.actions: list[action.ActionAcceleration] = self.actions
        self._state: state.StatePositionVelocity = self._state
        self._action: action.ActionAcceleration = self._action

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

    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position: common.XY = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                for vx in range(self.min_vx, self.max_vx+1):
                    for vy in range(self.min_vy, self.max_vy+1):
                        new_state: state.StatePositionVelocity = state.StatePositionVelocity(
                            is_terminal=is_terminal,
                            position=position,
                            velocity=common.XY(x=vx, y=vy)
                        )
                        self.states.append(new_state)

    def _build_actions(self):
        for ax in range(self.min_ax, self.max_ax + 1):
            for ay in range(self.min_ay, self.max_ay + 1):
                new_action: action.ActionAcceleration = action.ActionAcceleration(
                    acceleration=common.XY(x=ax, y=ay)
                )
                self.actions.append(new_action)

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=-1.0,
            state=self._projected_state
        )
