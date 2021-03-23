from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
from mdp.model.environment import Response
from mdp.scenarios.position_move.model import environment

from mdp.scenarios.windy.grid_world import GridWorld


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world: GridWorld = GridWorld(environment_parameters)

    def _get_response(self) -> Response:
        return Response(
            reward=-1.0,
            state=self._new_state
        )
