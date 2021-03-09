from __future__ import annotations

import random
import sys

from mdp.model import environment
from mdp.scenarios.jacks import state, action, environment_parameters, location, location_outcome   # grid_world


class Environment(environment.Environment):
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        # grid_world_ = grid_world.GridWorld(environment_parameters_)
        super().__init__(environment_parameters_, grid_world_=None)
        # super().__init__(environment_parameters_, grid_world_)

        # downcast states and actions so properties can be used freely
        self.states: list[state.State] = self.states
        self.actions: list[action.Action] = self.actions
        self._state: state.State = self._state
        self._action: action.Action = self._action

        self.dynamics = environment.Dynamics()

        self._max_cars: int = environment_parameters_.max_cars
        self._max_transfers: int = environment_parameters_.max_transfers
        self._rental_revenue: float = environment_parameters_.rental_revenue
        self._transfer_cost: float = environment_parameters_.transfer_cost
        self._extra_rules: bool = environment_parameters_.extra_rules

        self._location_1: location.Location = location.Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters_.rental_rate_1,
            return_rate=environment_parameters_.return_rate_1,
            excess_parking_cost=environment_parameters_.excess_parking_cost,
        )
        self._location_2: location.Location = location.Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters_.rental_rate_2,
            return_rate=environment_parameters_.return_rate_2,
            excess_parking_cost=environment_parameters_.excess_parking_cost,
        )

        self.counter: int = 0

    # region Sets
    def _build_states(self):
        """set S"""
        for cars1 in range(self._max_cars+1):
            for cars2 in range(self._max_cars+1):
                new_state: state.State = state.State(
                    cars_cob_1=cars1,
                    cars_cob_2=cars2,
                    is_terminal=False,
                )
                self.states.append(new_state)

    def _build_actions(self):
        for cars in range(-self._max_transfers, self._max_transfers+1):
            new_action: action.Action = action.Action(
                transfer_1_to_2=cars
            )
            self.actions.append(new_action)

    def is_action_compatible_with_state(self, state_: state.State, action_: action.Action):
        cars_sob_1 = state_.cars_cob_1 - action_.transfer_1_to_2
        cars_sob_2 = state_.cars_cob_2 + action_.transfer_1_to_2
        if 0 <= cars_sob_1 <= self._max_cars and \
                0 <= cars_sob_2 <= self._max_cars:
            return True
        else:
            return False
    # endregion

    # region Dynamics
    def _build_dynamics(self):
        for state_ in self.states:
            print(f"state = {state_}")
            print(f"cum dynamics entries = {self.counter}")
            for action_ in self.actions_for_state(state_):
                # print(state_, action_)
                self._add_dynamics(state_, action_)
        print(f"total dynamics entries = {self.counter}")
        sys.exit()

    def _add_dynamics(self, state_: state.State, action_: action.Action):
        total_costs: float = self._calc_cost_of_transfers(action_.transfer_1_to_2)
        if self._extra_rules:
            total_costs += self._location_1.parking_costs(state_.cars_cob_1)
            total_costs += self._location_2.parking_costs(state_.cars_cob_2)
        cars_sob_1 = state_.cars_cob_1 - action_.transfer_1_to_2
        cars_sob_2 = state_.cars_cob_2 + action_.transfer_1_to_2

        outcomes_1: list[location_outcome.LocationOutcome] = self._location_1.daily_outcomes[cars_sob_1]
        outcomes_2: list[location_outcome.LocationOutcome] = self._location_2.daily_outcomes[cars_sob_2]
        for outcome_1 in outcomes_1:
            for outcome_2 in outcomes_2:
                new_state = state.State(cars_cob_1=outcome_1.ending_cars,
                                        cars_cob_2=outcome_2.ending_cars,
                                        is_terminal=False)
                joint_probability = outcome_1.probability * outcome_2.probability
                total_cars_rented = outcome_1.cars_rented + outcome_2.cars_rented
                total_revenue = total_cars_rented * self._rental_revenue
                reward = total_revenue - total_costs
                probability_x_reward = joint_probability * reward
                self.counter += 1
                # self.dynamics.add(state_, action_, new_state, reward, joint_probability)

    def _calc_cost_of_transfers(self, transfer_1_to_2: int) -> float:
        """This will change for second part of problem"""
        if self._extra_rules and transfer_1_to_2 >= 1:
            transfers_incurring_cost = transfer_1_to_2 - 1  # one free ride
        else:
            transfers_incurring_cost = abs(transfer_1_to_2)
        transfer_cost = self._transfer_cost * transfers_incurring_cost
        return transfer_cost
    # endregion

    # region Operation
    def _get_a_start_state(self) -> state.State:
        return random.choice(self.states)

    def _apply_action(self):
        self._new_state, self._reward = self.dynamics.draw()

    def _get_response(self) -> environment.Response:
        return environment.Response(
            reward=self._reward,
            state=self._new_state
        )
    # endregion
