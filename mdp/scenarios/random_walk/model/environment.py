from __future__ import annotations

from mdp import common

from mdp.scenarios.position_move.model.state import State
from mdp.scenarios.position_move.model import environment

from mdp.scenarios.random_walk.model.dynamics import Dynamics
from mdp.scenarios.random_walk.model.grid_world import GridWorld


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics = Dynamics(environment_=self, environment_parameters=self._environment_parameters)

    def get_optimum(self, state: State) -> float:
        return self.grid_world.get_optimum(state.position)
