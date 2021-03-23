from __future__ import annotations
from typing import TYPE_CHECKING

import random
import numpy as np

if TYPE_CHECKING:
    from mdp.model import algorithm, policy
    from mdp.model.algorithm.value_function import state_function

from mdp import common
from mdp.model import environment

from mdp.scenarios.jacks.state import State
from mdp.scenarios.jacks.action import Action
from mdp.scenarios.jacks.environment_parameters import EnvironmentParameters
from mdp.scenarios.jacks.grid_world import GridWorld
from mdp.scenarios.jacks.dynamics import Dynamics


class Environment(environment.Environment):
    def __init__(self, environment_parameters: EnvironmentParameters):

        super().__init__(environment_parameters)

        # super().__init__(environment_parameters_, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[State] = self.states
        self.actions: list[Action] = self.actions
        self._state: State = self._state
        self._action: Action = self._action

        self.dynamics: Dynamics = Dynamics(environment_=self, environment_parameters=environment_parameters)
        self.grid_world: GridWorld = GridWorld(environment_parameters)

        self._max_cars: int = environment_parameters.max_cars
        self._max_transfers: int = environment_parameters.max_transfers
        # self._rental_revenue: float = environment_parameters_.rental_revenue
        # self._transfer_cost: float = environment_parameters_.transfer_cost
        # self._extra_rules: bool = environment_parameters_.extra_rules

        # self._location_1: Location = Location(
        #     max_cars=self._max_cars,
        #     rental_rate=environment_parameters_.rental_rate_1,
        #     return_rate=environment_parameters_.return_rate_1,
        #     excess_parking_cost=environment_parameters_.excess_parking_cost,
        # )
        # self._location_2: Location = Location(
        #     max_cars=self._max_cars,
        #     rental_rate=environment_parameters_.rental_rate_2,
        #     return_rate=environment_parameters_.return_rate_2,
        #     excess_parking_cost=environment_parameters_.excess_parking_cost,
        # )
        #
        # self.counter: int = 0

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

    # region Dynamics
    # def _add_dynamics(self, state_: State, action_: Action):
    #     total_costs: float = self._calc_cost_of_transfers(action_.transfer_1_to_2)
    #     if self._extra_rules:
    #         total_costs += self._location_1.parking_costs(state_.ending_cars_1)
    #         total_costs += self._location_2.parking_costs(state_.ending_cars_2)
    #
    #     cars_sob_1 = state_.ending_cars_1 - action_.transfer_1_to_2
    #     cars_sob_2 = state_.ending_cars_2 + action_.transfer_1_to_2
    #
    #     outcomes_1: list[LocationOutcome] = self._location_1.outcome_lookup[cars_sob_1]
    #     outcomes_2: list[LocationOutcome] = self._location_2.outcome_lookup[cars_sob_2]
    #
    #     # calculated expected reward given s,a
    #     expected_cars_rented_1: float = sum(outcome.probability_x_cars_rented for outcome in outcomes_1)
    #     expected_cars_rented_2: float = sum(outcome.probability_x_cars_rented for outcome in outcomes_2)
    #     expected_cars_rented = expected_cars_rented_1 + expected_cars_rented_2
    #     expected_revenue = expected_cars_rented * self._rental_revenue
    #     # sum_over_s'_r( p(s',r|s,a) . r )
    #     expected_reward = expected_revenue - total_costs
    #     self.dynamics.set_expected_reward(state_, action_, expected_reward)
    #     # self._expected_reward[(state_, action_)] = expected_reward
    #
    #     for outcome_1 in outcomes_1:
    #         for outcome_2 in outcomes_2:
    #             # p(s'|s,a) = p(s1'|s1,a).p(s2'|s2,a)
    #             probability = outcome_1.probability * outcome_2.probability
    #
    #             new_state = State(cars_cob_1=outcome_1.ending_cars,
    #                               cars_cob_2=outcome_2.ending_cars,
    #                               is_terminal=False)
    #             self.dynamics.set_next_state_probability(state_, action_, new_state, probability)
    #             self.counter += 1
    # state, action reaches new_state with probability
    # self.dynamics.add(state_, action_, new_state, probability)
    # endregion

    # region Operation
    def initialize_policy(self, policy_: policy.Policy, policy_parameters: common.PolicyParameters):
        initial_action = Action(transfer_1_to_2=0)
        for state in self.states:
            # max_transfer = min(state.cars_cob_1, self._max_cars - state.cars_cob_2, self._max_transfers)
            # max_transfer = -min(state.cars_cob_2, self._max_cars - state.cars_cob_1, self._max_transfers)
            # initial_action = Action(transfer_1_to_2=max_transfer)
            policy_[state] = initial_action
            # print(state, initial_action)

    def _get_a_start_state(self) -> State:
        return random.choice(self.states)

    def insert_state_function_into_graph3d(self, comparison: common.Comparison, v: state_function.StateFunction):
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

    def update_grid_value_functions(self, algorithm_: algorithm.Algorithm, policy_: policy.Policy):
        # policy_: policy.Deterministic
        for state in self.states:
            position: common.XY = common.XY(x=state.ending_cars_2, y=state.ending_cars_1)     # reversed like in book
            action: Action = policy_[state]
            transfer_1_to_2: int = action.transfer_1_to_2
            # print(position, transfer_1_to_2)
            self.grid_world.set_policy_value(
                position=position,
                policy_value=transfer_1_to_2,
            )
            # if algorithm_.Q:
            #     policy_action: Optional[environment.Action] = policy_[state]
            #     policy_action: Action
            #     policy_move: Optional[common.XY] = None
            #     if policy_action:
            #         policy_move = policy_action.move
            #     for action_ in self.actions_for_state(state):
            #         is_policy: bool = (policy_move and policy_move == action_.move)
            #         self.grid_world.set_state_action_function(
            #             position=state.position,
            #             move=action_.move,
            #             q_value=algorithm_.Q[state, action_],
            #             is_policy=is_policy
            #         )
        # print(self.grid_world.output_squares)

    def is_valued_state(self, state: State) -> bool:
        return True
    # endregion
