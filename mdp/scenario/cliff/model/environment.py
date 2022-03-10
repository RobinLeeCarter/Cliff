from __future__ import annotations


from mdp import common
from mdp.scenario.position_move.model.state import State
from mdp.scenario.position_move.model.grid_world import GridWorld
from mdp.scenario.position_move.model import environment
from mdp.scenario.cliff.model.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world = GridWorld(self._environment_parameters)
        self.dynamics = Dynamics(environment=self, environment_parameters=self._environment_parameters)

    def is_valued_state(self, state: State) -> bool:
        _square: int = self.grid_world.get_square(state.position)
        if _square in (common.Square.END, common.Square.CLIFF):
            return False
        else:
            return True
