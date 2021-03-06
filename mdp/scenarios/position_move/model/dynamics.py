from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC

if TYPE_CHECKING:
    from mdp.scenarios.position_move.model.action import Action
    from mdp.scenarios.position_move.model.environment import Environment
    from mdp.scenarios.position_move.model.grid_world import GridWorld
    from mdp.common import EnvironmentParameters

from mdp import common
from mdp.model.environment import dynamics

from mdp.scenarios.position_move.model.state import State


class Dynamics(dynamics.Dynamics, ABC):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment
        self._grid_world: GridWorld = self._environment.grid_world

        # current values
        self._next_state: Optional[State] = None
        self._square: Optional[int] = None

    def get_start_states(self) -> list[State]:
        start_positions: list[common.XY] = self._grid_world.get_start_positions()
        start_states = [State(is_terminal=False, position=position)
                        for position in start_positions]
        return start_states

    def _draw_next_state(self, state: State, action: Action):
        move: Optional[common.XY] = None

        if action:
            move = action.move
        new_position = self._grid_world.change_request(
            current_position=state.position,
            move=move)

        self._square = self._grid_world.get_square(new_position)
        if self._square == common.Square.END:
            is_terminal = True
        else:
            is_terminal = False
        self._next_state = State(is_terminal, new_position)
