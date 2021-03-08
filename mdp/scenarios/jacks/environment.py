from __future__ import annotations

from mdp import common
from mdp.model import environment
from mdp.scenarios.jacks import grid_world, state


class Environment(environment.Environment):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        grid_world_ = grid_world.GridWorld(environment_parameters)
        super().__init__(environment_parameters, grid_world_)
        self.dynamics = environment.Dynamics()

    def _get_response(self) -> environment.Response:
        reward: float
        self._new_state: state.State
        if self._new_state.position == common.XY(x=self.grid_world.max_x, y=0):
            reward = 1.0
        else:
            reward = 0.0

        return environment.Response(
            reward=reward,
            state=self._new_state
        )

    def get_optimum(self, state_: environment.State) -> float:
        self.grid_world: grid_world.GridWorld
        state_: state.State
        return self.grid_world.get_optimum(state_.position)
