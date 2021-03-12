from __future__ import annotations

from mdp import common

from mdp.model.environment import Response
# from mdp.scenarios.position_move.model.state import State
# from mdp.scenarios.position_move.model import Environment as PositionMoveEnvironment, State
from mdp.scenarios.position_move.model import environment, State

# from mdp.model import environment as base_environment
# from mdp.scenarios.position_move import environment, state

from mdp.scenarios.random_walk.grid_world import GridWorld


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_world_ = GridWorld(environment_parameters)
        super().__init__(environment_parameters, grid_world_)

    def _get_response(self) -> Response:
        reward: float
        self._new_state: State
        if self._new_state.position == common.XY(x=self.grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return Response(
            reward=reward,
            state=self._new_state
        )

    def get_optimum(self, state: State) -> float:
        self.grid_world: GridWorld
        state: State
        return self.grid_world.get_optimum(state.position)
