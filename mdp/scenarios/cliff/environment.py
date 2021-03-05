from __future__ import annotations


from mdp import common
from mdp.model import environment as base_environment
from mdp.scenarios.position_move import environment, grid_world


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_world_ = grid_world.GridWorld(environment_parameters)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> base_environment.Response:
        if self._square == common.Square.CLIFF:
            return base_environment.Response(
                reward=-100.0,
                state=self._get_a_start_state()
            )
        else:
            return base_environment.Response(
                reward=-1.0,
                state=self._new_state
            )

