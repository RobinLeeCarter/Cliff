from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model import environment as base_environment
from mdp.scenarios.position_move import environment
from mdp.scenarios.windy import grid_world


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_world_ = grid_world.GridWorld(environment_parameters)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> base_environment.Response:
        return base_environment.Response(
            reward=-1.0,
            state=self._new_state
        )
