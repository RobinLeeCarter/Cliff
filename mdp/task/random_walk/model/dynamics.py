from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.common import EnvironmentParameters
    from mdp.task._position_move.model.action import Action
    from mdp.task.random_walk.model.environment import Environment
    from mdp.task.random_walk.model.grid_world import GridWorld


from mdp import common

from mdp.task._position_move.model.state import State
from mdp.task._position_move.model import dynamics


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment, environment_parameters)
        self._environment: Environment = environment
        self._grid_world: GridWorld = self._environment.grid_world

    def draw_response(self, state: State, action: Action) -> tuple[float, State]:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        self._draw_next_state(state, action)

        if self._next_state.position == common.XY(x=self._grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return reward, self._next_state
