from __future__ import annotations

import random

from mdp.model import environment
from mdp.scenarios.jacks import state, action, environment_parameters, location, location_outcome   # grid_world


class Environment(environment.Environment):
    def __init__(self, environment_parameters_: environment_parameters.EnvironmentParameters):
        # grid_world_ = grid_world.GridWorld(environment_parameters_)
        super().__init__(environment_parameters_, grid_world_=None)
        # super().__init__(environment_parameters_, grid_world_)
        self.dynamics = environment.Dynamics()

        # downcast states and actions so properties can be used freely
        self.states: list[state.State] = self.states
        self.actions: list[action.Action] = self.actions
        self._state: state.State = self._state
        self._action: action.Action = self._action

        self._max_cars: int = environment_parameters_.max_cars
        self._max_transfers: int = environment_parameters_.max_transfers
        self._rental_revenue: float = environment_parameters_.rental_revenue
        self._transfer_cost: float = environment_parameters_.transfer_cost

        self._location_1: location.Location = location.Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters_.rental_rate_1,
            return_rate=environment_parameters_.return_rate_1,
        )
        self._location_2: location.Location = location.Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters_.rental_rate_2,
            return_rate=environment_parameters_.return_rate_2,
        )

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
            for action_ in self.actions_for_state(state_):
                self._add_dynamics(state_, action_)

    def _add_dynamics(self, state_: state.State, action_: action.Action):
        total_costs: float = 0.0
        total_costs += self._calc_cost_of_transfers(action_.transfer_1_to_2)
        cars_sob_1 = state_.cars_cob_1 - action_.transfer_1_to_2
        cars_sob_2 = state_.cars_cob_2 + action_.transfer_1_to_2

        location_outcome_1: location_outcome.LocationOutcome
        location_outcome_2: location_outcome.LocationOutcome
        for location_outcome_1 in self._location_1.day_distribution(cars_sob_1):
            for location_outcome_2 in self._location_2.day_distribution(cars_sob_2):
                total_cars_rented = location_outcome_1.cars_rented + location_outcome_2.cars_rented
                total_revenue = total_cars_rented * self._rental_revenue
                joint_probability = location_outcome_1.probability * location_outcome_2.probability
                reward = total_revenue - total_costs
                new_state = state.State(cars_cob_1=location_outcome_1.ending_cars,
                                        cars_cob_2=location_outcome_2.ending_cars,
                                        is_terminal=False)
                self.dynamics.add(state_, action_, new_state, reward, joint_probability)

    def _calc_cost_of_transfers(self, transfer_1_to_2: int) -> float:
        """This will change for second part of problem"""
        transfer_cost = self._transfer_cost * abs(transfer_1_to_2)
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
