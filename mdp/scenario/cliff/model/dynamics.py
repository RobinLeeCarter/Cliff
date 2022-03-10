from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.common import EnvironmentParameters
    from mdp.scenario.position_move.model.action import Action
    from mdp.scenario.cliff.model.environment import Environment


from mdp import common

from mdp.scenario.position_move.model.state import State
from mdp.scenario.position_move.model import dynamics


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment, environment_parameters)

        # downcast
        self._environment: Environment = self._environment

    def draw_response(self, state: State, action: Action) -> tuple[float, Optional[State]]:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        self._draw_next_state(state, action)

        reward: float
        new_state: Optional[State]
        if self._square == common.Square.CLIFF:
            reward = -100.0
            new_state = None    # start state
        else:
            reward = -1.0
            new_state = self._next_state
        return reward, new_state
