from __future__ import annotations
from abc import ABC
from typing import Optional

from mdp import common
from mdp.model import environment
# from mdp.model.environment import grid_world
from mdp.model.scenarios.common import state_position, actions_factory, action_move


class Environment(environment.Environment, ABC):
    # def __init__(self,
    #              environment_parameters: common.EnvironmentParameters,
    #              grid_world_: grid_world.GridWorld):
    #     self.__init__(environment_parameters, grid_world_)

    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                new_state: state_position.State = state_position.State(
                    position=position,
                    is_terminal=is_terminal,
                )
                self.states.append(new_state)

    def _build_actions(self):
        self.actions = actions_factory.ActionsFactory(actions_list=self._environment_parameters.actions_list)

    def _get_a_start_state(self) -> state_position.State:
        position: common.XY = self.grid_world.get_a_start_position()
        return state_position.State(is_terminal=False, position=position)

    def _apply_action(self):
        # apply grid world rules (eg. edges, wind)
        self._state: state_position.State
        self._action: action_move.Action

        move: Optional[common.XY] = None
        if self._action:
            move = self._action.move
        self._projected_position = self.grid_world.change_request(
            current_position=self._state.position,
            move=move)

        self._square = self.grid_world.get_square(self._projected_position)
        if self._square == common.Square.END:
            is_terminal = True
        else:
            is_terminal = False
        self._projected_state = state_position.State(is_terminal, self._projected_position)
