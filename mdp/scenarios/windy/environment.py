from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model import environment
from mdp.scenarios.common.model import position_move
from mdp.scenarios.windy import grid_world


class Environment(position_move.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_world_ = grid_world.GridWorld(environment_parameters)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=-1.0,
            state=self._new_state
        )
