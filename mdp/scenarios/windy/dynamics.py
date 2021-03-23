from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.common import EnvironmentParameters
    from mdp.scenarios.position_move.model.action import Action
    from mdp.scenarios.windy.environment import Environment
    from mdp.scenarios.windy.grid_world import GridWorld

from mdp.scenarios.position_move.model import State, Response, dynamics


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._grid_world: GridWorld = self._environment.grid_world

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        self._draw_next_state(state, action)

        return Response(reward=-1.0, state=self._next_state)
