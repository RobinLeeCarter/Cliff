from __future__ import annotations
from typing import Optional

from mdp import common
from mdp.model.environment import environment

from mdp.scenarios.racetrack.model.state import State
from mdp.scenarios.racetrack.model.action import Action
from mdp.scenarios.racetrack.model.grid_world import GridWorld
from mdp.scenarios.racetrack.model.dynamics import Dynamics
from mdp.scenarios.racetrack.model.environment_parameters import EnvironmentParameters


class Environment(environment.Environment):
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters)

        # downcast states and actions so properties can be used freely
        self.states: list[State] = self.states
        self.actions: list[Action] = self.actions
        # self._state: State = self._state
        # self._action: Action = self._action
        self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics: Dynamics = Dynamics(environment_=self, environment_parameters=environment_parameters)

        self._reward: float = 0.0
        self._next_state: Optional[State] = None

        # velocity
        self._min_vx: int = environment_parameters.min_velocity
        self._max_vx: int = environment_parameters.max_velocity
        self._min_vy: int = environment_parameters.min_velocity
        self._max_vy: int = environment_parameters.max_velocity

        # acceleration
        self._min_ax: int = environment_parameters.min_acceleration
        self._max_ax: int = environment_parameters.max_acceleration
        self._min_ay: int = environment_parameters.min_acceleration
        self._max_ay: int = environment_parameters.max_acceleration

    # region Sets
    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position: common.XY = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                for vx in range(self._min_vx, self._max_vx + 1):
                    for vy in range(self._min_vy, self._max_vy + 1):
                        new_state: State = State(
                            is_terminal=is_terminal,
                            position=position,
                            velocity=common.XY(x=vx, y=vy)
                        )
                        self.states.append(new_state)

    def _build_actions(self):
        for ax in range(self._min_ax, self._max_ax + 1):
            for ay in range(self._min_ay, self._max_ay + 1):
                new_action: Action = Action(
                    acceleration=common.XY(x=ax, y=ay)
                )
                self.actions.append(new_action)

    def is_action_compatible_with_state(self, state: State, action: Action):
        new_vx = state.velocity.x + action.acceleration.x
        new_vy = state.velocity.y + action.acceleration.y
        if self._min_vx <= new_vx <= self._max_vx and \
            self._min_vy <= new_vy <= self._max_vy and \
                not (new_vx == 0 and new_vy == 0):
            return True
        else:
            return False
    # endregion

    # region Operation
    def output_mode(self):
        self.grid_world.skid_probability = 0.0
    # endregion
