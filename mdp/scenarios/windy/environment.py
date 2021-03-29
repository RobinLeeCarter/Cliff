from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.scenarios.position_move import environment

from mdp.scenarios.windy.grid_world import GridWorld
from mdp.scenarios.windy.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics = Dynamics(environment_=self, environment_parameters=self._environment_parameters)
