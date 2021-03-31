from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.common import EnvironmentParameters
    from mdp.scenarios.position_move.model.action import Action
    from mdp.scenarios.cliff.model.environment import Environment


from mdp import common

from mdp.scenarios.position_move.model.state import State
from mdp.scenarios.position_move.model.response import Response
from mdp.scenarios.position_move.model import dynamics


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        self._draw_next_state(state, action)

        if self._square == common.Square.CLIFF:
            response = Response(
                reward=-100.0,
                state=self.get_a_start_state()
            )
        else:
            response = Response(
                reward=-1.0,
                state=self._next_state
            )
        return response
