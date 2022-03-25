from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.task.racetrack.model.environment import Environment
    from mdp.task.racetrack.model.environment_parameters import EnvironmentParameters
    from mdp.task.racetrack.model.grid_world import GridWorld

from mdp import common
from mdp.task.racetrack.model.state import State
from mdp.task.racetrack.model.action import Action

from mdp.model.tabular.environment.tabular_dynamics import TabularDynamics


class Dynamics(TabularDynamics[State, Action]):
    def __init__(self, environment: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment, environment_parameters)
        self._environment: Environment = environment
        self._environment_parameters: EnvironmentParameters = environment_parameters
        self._grid_world: GridWorld = self._environment.grid_world  # downcast

        self._extra_reward_for_failure: float = environment_parameters.extra_reward_for_failure

    def get_start_states(self) -> list[State]:
        start_positions: list[common.XY] = self._grid_world.get_start_positions()
        start_velocity = common.XY(x=0, y=0)
        start_states = [State(is_terminal=False, position=position, velocity=start_velocity)
                        for position in start_positions]
        return start_states

    # @profile
    def draw_response(self, state: State, action: Action) -> tuple[float, Optional[State]]:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        :returns state = None if a new starting state required as this is more efficient if want s not state
        """
        new_position, new_velocity = self._grid_world.change_request(
            position=state.position,
            velocity=state.velocity,
            acceleration=action.acceleration
            )
        square: int = self._grid_world.get_square(new_position)

        reward: float
        next_state: Optional[State]
        if square == common.Square.END:
            # success
            reward = 0.0
            new_position = self._grid_world.project_back_to_grid(new_position)
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
            next_state = None
            # s: int = self._environment.start_s_distribution.draw_one()
            # next_state: State = self._environment.states[s]
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
        return reward, next_state
