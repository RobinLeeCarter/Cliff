from __future__ import annotations
from abc import ABC
from typing import Optional, TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from mdp.model import algorithm, policy

from mdp import common
from mdp.model import environment
from mdp.model.environment import grid_world as mdp_model_grid_world
from mdp.scenarios.common.model.position_move import actions_factory, action, grid_world
from mdp.scenarios.common.model.position_move import state


class EnvironmentStatePosition(environment.Environment, ABC):
    def __init__(self,
                 environment_parameters: common.EnvironmentParameters,
                 grid_world_: mdp_model_grid_world.GridWorld):
        super().__init__(environment_parameters, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[state.StatePosition] = self.states
        self.actions: list[action.ActionMove] = self.actions
        self._state: state.StatePosition = self._state
        self._action: action.ActionMove = self._action
        self.grid_world: grid_world.GridWorldPositionMove = self.grid_world

    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                new_state: state.StatePosition = state.StatePosition(
                    position=position,
                    is_terminal=is_terminal,
                )
                self.states.append(new_state)

    def _build_actions(self):
        self.actions = actions_factory.factory(actions_list=self._environment_parameters.actions_list)

    # noinspection PyUnusedLocal
    def actions_for_state(self, state_: state.StatePosition) -> Generator[action.ActionMove, None, None]:
        """set A(s)"""
        for action_ in self.actions:
            # if self.is_action_compatible_with_state(state_, action_):
            yield action_

    def _get_a_start_state(self) -> state.StatePosition:
        position: common.XY = self.grid_world.get_a_start_position()
        return state.StatePosition(is_terminal=False, position=position)

    def _apply_action(self):
        # apply grid world rules (eg. edges, wind)
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
        self._projected_state = state.StatePosition(is_terminal, self._projected_position)

    def update_grid_value_functions(self, algorithm_: algorithm.Episodic, policy_: policy.Policy):
        for state_ in self.states:
            if algorithm_.V:
                self.grid_world.set_state_function(
                    position=state_.position,
                    v_value=algorithm_.V[state_]
                )
            if algorithm_.Q:
                policy_action: Optional[environment.Action] = policy_[state_]
                policy_action: action.ActionMove
                policy_move: Optional[common.XY] = None
                if policy_action:
                    policy_move = policy_action.move
                for action_ in self.actions_for_state(state_):
                    is_policy: bool = (policy_move and policy_move == action_.move)
                    self.grid_world.set_state_action_function(
                        position=state_.position,
                        move=action_.move,
                        q_value=algorithm_.Q[state_, action_],
                        is_policy=is_policy
                    )

    def is_valued_state(self, state_: state.StatePosition) -> bool:
        _square: common.Square = self.grid_world.get_square(state_.position)
        square_enum = common.enums.Square
        if _square in (square_enum.END, square_enum.CLIFF):
            return False
        else:
            return True
