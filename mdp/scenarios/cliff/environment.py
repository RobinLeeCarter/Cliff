from __future__ import annotations


from mdp import common
from mdp.model.environment import Response
from mdp.scenarios.position_move.model import environment, GridWorld


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)
        self.grid_world: GridWorld = GridWorld(environment_parameters)

    def _get_response(self) -> Response:
        if self._square == common.Square.CLIFF:
            return Response(
                reward=-100.0,
                state=self._get_a_start_state()
            )
        else:
            return Response(
                reward=-1.0,
                state=self._new_state
            )
