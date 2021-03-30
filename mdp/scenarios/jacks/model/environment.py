from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mdp.model.algorithm.abstract.algorithm_ import Algorithm
    from mdp.model.policy.policy import Policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model.environment import environment

from mdp.scenarios.jacks.model.state import State
from mdp.scenarios.jacks.model.action import Action
from mdp.scenarios.jacks.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.jacks.model.grid_world import GridWorld
from mdp.scenarios.jacks.dynamics.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: EnvironmentParameters):

        super().__init__(environment_parameters)

        # super().__init__(environment_parameters_, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[State] = self.states
        self.actions: list[Action] = self.actions
        self._state: State = self._state
        self._action: Action = self._action

        self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics: Dynamics = Dynamics(environment_=self, environment_parameters=environment_parameters)

        self._max_cars: int = environment_parameters.max_cars
        self._max_transfers: int = environment_parameters.max_transfers

    # region Sets
    def _build_states(self):
        """set S"""
        for cars1 in range(self._max_cars+1):
            for cars2 in range(self._max_cars+1):
                new_state: State = State(
                    ending_cars_1=cars1,
                    ending_cars_2=cars2,
                    is_terminal=False,
                )
                self.states.append(new_state)

    def _build_actions(self):
        for cars in range(-self._max_transfers, self._max_transfers+1):
            new_action: Action = Action(
                transfer_1_to_2=cars
            )
            self.actions.append(new_action)

    def is_action_compatible_with_state(self, state_: State, action_: Action):
        starting_cars_1 = state_.ending_cars_1 - action_.transfer_1_to_2
        starting_cars_2 = state_.ending_cars_2 + action_.transfer_1_to_2
        if 0 <= starting_cars_1 <= self._max_cars and \
                0 <= starting_cars_2 <= self._max_cars:
            return True
        else:
            return False
    # endregion

    # region Operation
    def initialize_policy(self, policy_: Policy, policy_parameters: common.PolicyParameters):
        initial_action = Action(transfer_1_to_2=0)
        for state in self.states:
            # max_transfer = min(state.cars_cob_1, self._max_cars - state.cars_cob_2, self._max_transfers)
            # max_transfer = -min(state.cars_cob_2, self._max_cars - state.cars_cob_1, self._max_transfers)
            # initial_action = Action(transfer_1_to_2=max_transfer)
            policy_[state] = initial_action
            # print(state, initial_action)

    def insert_state_function_into_graph3d(self,
                                           comparison: common.Comparison,
                                           v: state_function.StateFunction,
                                           parameter: Optional[any] = None):
        x_values = np.arange(self._max_cars + 1, dtype=float)
        y_values = np.arange(self._max_cars + 1, dtype=float)
        z_values = np.empty(shape=(self._max_cars + 1, self._max_cars + 1), dtype=float)

        for cars1 in range(self._max_cars+1):
            for cars2 in range(self._max_cars+1):
                state: State = State(
                    ending_cars_1=cars1,
                    ending_cars_2=cars2,
                    is_terminal=False,
                )
                z_values[cars2, cars1] = v[state]
                # print(cars1, cars2, v[state])

        g = comparison.graph3d_values
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.y_series = common.Series(title=g.y_label, values=y_values)
        g.z_series = common.Series(title=g.z_label, values=z_values)

    def update_grid_policy(self, policy: Policy):
        # policy_: policy.Deterministic
        for state in self.states:
            position: common.XY = common.XY(x=state.ending_cars_2, y=state.ending_cars_1)     # reversed like in book
            action: Action = policy[state]
            transfer_1_to_2: int = action.transfer_1_to_2
            # print(position, transfer_1_to_2)
            self.grid_world.set_policy_value(
                position=position,
                policy_value=transfer_1_to_2,
            )

    def is_valued_state(self, state: State) -> bool:
        return True
    # endregion
