from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.racetrack.model.action import Action
    from mdp.scenarios.racetrack.model.environment import Environment
    from mdp.scenarios.racetrack.model.environment_parameters import EnvironmentParameters
    from mdp.scenarios.racetrack.model.grid_world import GridWorld

from mdp import common
from mdp.model.environment import dynamics

from mdp.scenarios.racetrack.model.state import State
from mdp.scenarios.racetrack.model.response import Response


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._grid_world: GridWorld = self._environment.grid_world

        self._extra_reward_for_failure: float = environment_parameters.extra_reward_for_failure

    def get_a_start_state(self) -> State:
        position: common.XY = self._grid_world.get_a_start_position()
        return State(is_terminal=False, position=position, velocity=common.XY(x=0, y=0))

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        new_position, new_velocity = self._grid_world.change_request(
            position=state.position,
            velocity=state.velocity,
            acceleration=action.acceleration
            )
        square: common.Square = self._grid_world.get_square(new_position)

        reward: float
        next_state: State
        if square == common.Square.END:
            # success
            reward = 0.0
            next_state = State(
                position=new_position,
                velocity=new_velocity,
                is_terminal=True
            )
            if self._verbose:
                print(f"Past finish line at {new_position}")
        elif square == common.Square.CLIFF:
            # failure, move back to start line
            # self.pre_reset_state = State(x, y, vx, vy, is_reset=True)
            reward: float = -1.0 + self._extra_reward_for_failure
            next_state: State = self.get_a_start_state()
            if self._verbose:
                print(f"Grass at {new_position}")
        else:
            # TRACK or START so continue
            reward = -1.0
            next_state = State(
                position=new_position,
                velocity=new_velocity,
                is_terminal=False
            )
        return Response(reward, next_state)
