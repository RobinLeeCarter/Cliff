from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from mdp.model import algorithm, policy

from mdp import common
from mdp.model import environment
from mdp.scenarios.factory import actions_list_factory

from mdp.scenarios.position_move.model import State, Action, GridWorld, Dynamics


class Environment(environment.Environment, abc.ABC):
    def __init__(self, environment_parameters: common.EnvironmentParameters):
        super().__init__(environment_parameters)

        # downcast states and actions so properties can be used freely
        self.states: list[State] = self.states
        self.actions: list[Action] = self.actions
        self._state: State = self._state
        self._action: Action = self._action
        self.grid_world: Optional[GridWorld] = None
        self.dynamics: Optional[Dynamics] = None

    # region Sets
    def _build_states(self):
        """set S"""
        for x in range(self.grid_world.max_x+1):
            for y in range(self.grid_world.max_y+1):
                position = common.XY(x=x, y=y)
                is_terminal: bool = self.grid_world.is_at_goal(position)
                new_state: State = State(
                    position=position,
                    is_terminal=is_terminal,
                )
                self.states.append(new_state)

    def _build_actions(self):
        self.actions = actions_list_factory.actions_list_factory(actions_list=self._environment_parameters.actions_list)

    def actions_for_state(self, state: State) -> Generator[Action, None, None]:
        """set A(s)"""
        for action_ in self.actions:
            if self.is_action_compatible_with_state(state, action_):
                yield action_
    # endregion

    # region Operation
    def update_grid_value_functions(self,
                                    algorithm_: algorithm.Algorithm,
                                    policy_: policy.Policy,
                                    parameter: any = None
                                    ):
        for state in self.states:
            if algorithm_.V:
                self.grid_world.set_v_value(
                    position=state.position,
                    v_value=algorithm_.V[state]
                )
            if algorithm_.Q:
                policy_action: Optional[environment.Action] = policy_[state]
                policy_action: Action
                policy_move: Optional[common.XY] = None
                if policy_action:
                    policy_move = policy_action.move
                for action_ in self.actions_for_state(state):
                    is_policy: bool = (policy_move and policy_move == action_.move)
                    self.grid_world.set_move_q_value(
                        position=state.position,
                        move=action_.move,
                        q_value=algorithm_.Q[state, action_],
                        is_policy=is_policy
                    )

    def is_valued_state(self, state: State) -> bool:
        _square: common.Square = self.grid_world.get_square(state.position)
        square_enum = common.enums.Square
        if _square in (square_enum.END, square_enum.CLIFF):
            return False
        else:
            return True
    # endregion
