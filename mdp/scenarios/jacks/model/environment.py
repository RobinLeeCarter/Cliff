from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.policy.tabular_policy import TabularPolicy
    from mdp.model.tabular.value_function import state_function

from mdp import common
from mdp.scenarios.jacks.model.state import State
from mdp.scenarios.jacks.model.action import Action
from mdp.scenarios.jacks.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.jacks.model.grid_world import GridWorld
from mdp.scenarios.jacks.model.dynamics.dynamics import Dynamics

from mdp.model.tabular.environment.tabular_environment import TabularEnvironment


class Environment(TabularEnvironment[State, Action]):
    def __init__(self, environment_parameters: EnvironmentParameters):
        super().__init__(environment_parameters)
        self._environment_parameters: EnvironmentParameters = environment_parameters

        self.grid_world: GridWorld = GridWorld(environment_parameters)
        self.dynamics: Dynamics = Dynamics(environment=self, environment_parameters=environment_parameters)

        self._max_cars: int = environment_parameters.max_cars
        self._max_transfers: int = environment_parameters.max_transfers

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

    def _is_action_compatible_with_state(self, state_: State, action_: Action):
        starting_cars_1 = state_.ending_cars_1 - action_.transfer_1_to_2
        starting_cars_2 = state_.ending_cars_2 + action_.transfer_1_to_2
        if 0 <= starting_cars_1 <= self._max_cars and \
                0 <= starting_cars_2 <= self._max_cars:
            return True
        else:
            return False

    def initialize_policy(self, policy: TabularPolicy):
        policy.zero_state_action()

        initial_action: Action = Action(transfer_1_to_2=0)
        initial_a: int = self.action_index[initial_action]
        for s, state in enumerate(self.states):
            # max_transfer = min(state.cars_cob_1, self._max_cars - state.cars_cob_2, self._max_transfers)
            # max_transfer = -min(state.cars_cob_2, self._max_cars - state.cars_cob_1, self._max_transfers)
            # initial_action = Action(transfer_1_to_2=max_transfer)
            policy[s] = initial_a
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
                s: int = self.state_index[state]
                z_values[cars2, cars1] = v[s]
                # print(cars1, cars2, v[state])

        g = comparison.graph3d_values
        g.x_series = common.Series(title=g.x_label, values=x_values)
        g.y_series = common.Series(title=g.y_label, values=y_values)
        g.z_series = common.Series(title=g.z_label, values=z_values)

    def update_grid_policy(self, policy: TabularPolicy):
        # policy_: policy.Deterministic
        for s, state in enumerate(self.states):
            position: common.XY = common.XY(x=state.ending_cars_2, y=state.ending_cars_1)     # reversed like in book
            action: Action = policy.get_action(s)   # type: ignore
            transfer_1_to_2: int = action.transfer_1_to_2
            # print(position, transfer_1_to_2)
            self.grid_world.set_policy_value(
                position=position,
                policy_value=transfer_1_to_2,
            )
