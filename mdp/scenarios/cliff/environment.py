from __future__ import annotations


from mdp import common
from mdp.scenarios.position_move.model.grid_world import GridWorld
from mdp.scenarios.position_move.model import environment
from mdp.scenarios.cliff.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world = GridWorld(self._environment_parameters)
        self.dynamics = Dynamics(environment_=self, environment_parameters=self._environment_parameters)
