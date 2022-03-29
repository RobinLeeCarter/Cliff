from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.task.position_move.model.state import State
    from mdp.task.random_walk.model.environment_parameters import EnvironmentParameters
from mdp.task.random_walk.model.dynamics import Dynamics
from mdp.task.random_walk.model.grid_world import GridWorld

from mdp.task.position_move.model import environment


class Environment(environment.Environment):
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters)
        self._environment_parameters: EnvironmentParameters = environment_parameters
        self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics = Dynamics(environment=self, environment_parameters=self._environment_parameters)

    def get_optimum(self, state: State) -> float:
        return self.grid_world.get_optimum(state.position)
