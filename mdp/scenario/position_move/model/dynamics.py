from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC

if TYPE_CHECKING:
    from mdp.scenario.position_move.model.environment import Environment
    from mdp.scenario.position_move.model.grid_world import GridWorld
    from mdp.common import EnvironmentParameters

from mdp import common
from mdp.scenario.position_move.model.state import State
from mdp.scenario.position_move.model.action import Action

from mdp.model.tabular.environment.tabular_dynamics import TabularDynamics


class Dynamics(TabularDynamics[State, Action], ABC):
    def __init__(self, environment: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment, environment_parameters)
        self._environment: Environment = environment
        self._environment_parameters: EnvironmentParameters = environment_parameters
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
