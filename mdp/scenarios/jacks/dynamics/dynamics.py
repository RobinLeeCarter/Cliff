from __future__ import annotations
from typing import TYPE_CHECKING, Optional

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
from mdp.scenarios.jacks.response import Response

from mdp.scenarios.jacks.dynamics.outcome import Outcome
from mdp.scenarios.jacks.dynamics.location import Location
from mdp.scenarios.jacks.dynamics.location_outcome import LocationOutcome
# from mdp.scenarios.jacks.dynamics.location_outcomes import LocationOutcomes
from mdp.scenarios.jacks.dynamics.dict_zero import DictZero
from mdp.scenarios.jacks.dynamics.state_probability import StateProbability


class Dynamics:
    def __init__(self, environment_parameters_: EnvironmentParameters):
        self.is_built: bool = False

        self._max_cars: int = environment_parameters_.max_cars
        self._max_transfers: int = environment_parameters_.max_transfers
        self._rental_revenue: float = environment_parameters_.rental_revenue
        self._transfer_cost: float = environment_parameters_.transfer_cost
        self._extra_rules: bool = environment_parameters_.extra_rules

        self._location_1: Location = Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters_.rental_rate_1,
            return_rate=environment_parameters_.return_rate_1,
            excess_parking_cost=environment_parameters_.excess_parking_cost,
        )
        self._location_2: Location = Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters_.rental_rate_2,
            return_rate=environment_parameters_.return_rate_2,
            excess_parking_cost=environment_parameters_.excess_parking_cost,
        )

    def build(self):
        pass

    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        r(s,a) = E[Rt | S(t-1)=s, A(t-1)=a] = Sum_over_s'_r( p(s',r|s,a).r )
        expected reward for a (state, action)
        """
        raise NotImplementedError

    def get_expected_conditional_reward(self, next_state: State, state: State, action: Action) -> float:
        """
        r(s,a,s') = E[Rt | S(t)=s', S(t-1)=s, A(t-1=a)] = Sum_over_r( p(s',r|s,a).r ) / p(s'|s,a)
        expected reward for a (state, action) given the next state
        """
        probability_x_reward = self.get_probability_x_reward(next_state, state, action)
        next_state_probability = self.get_next_state_probability(next_state, state, action)
        if next_state_probability == 0.0:
            return 0.0
        else:
            return probability_x_reward / next_state_probability

    def get_probability_x_reward(self, next_state: State, state: State, action: Action) -> float:
        """
        Sum_over_r( p(s',r|s,a).r )
        probability_x_reward for a (state, action) given the next state
        """
        raise NotImplementedError

    def get_next_state_probability(self, next_state: State, state: State, action: Action) -> float:
        """
        p(s'|s,a) = Sum_over_r( p(s',r|s,a) )
        probability of a next state for a (state, action)
        """
        raise NotImplementedError

    def get_next_state_distribution(self, state: State, action: Action) -> list[StateProbability]:
        """
        list[ s', p(s'|s,a) ]
        distribution of next states for a (state, action)
        """
        raise NotImplementedError

    def get_outcomes(self, state: State, action: Action) -> list[Outcome]:
        """
        list of possible outcomes for a single state and action
        could be used for one state, action in theory
        but too many for all states and actions so potentially not useful in practice
        """
        total_costs: float = self._calc_cost_of_transfers(action.transfer_1_to_2)
        if self._extra_rules:
            total_costs += self._location_1.parking_costs(state.cars_cob_1)
            total_costs += self._location_2.parking_costs(state.cars_cob_2)

        cars_sob_1 = state.cars_cob_1 - action.transfer_1_to_2
        cars_sob_2 = state.cars_cob_2 + action.transfer_1_to_2

        outcomes1: dict[LocationOutcome, float] = self._location_1.get_outcomes(cars_sob_1)
        outcomes2: dict[LocationOutcome, float] = self._location_2.get_outcomes(cars_sob_2)

        # 27,000 for one state and action, up to 120m for all states and actions
        # total_possibilities = len(outcomes1) * len(outcomes2)
        # print(f"total_possibilities = {total_possibilities}")

        # outcome_dict[(new_state, cars_rented)] = probability
        outcome_dict: DictZero[tuple[State, int], float] = DictZero()

        for outcome1, probability1 in outcomes1.items():
            for outcome2, probability2 in outcomes2.items():
                new_state = State(is_terminal=False, cars_cob_1=outcome1.ending_cars, cars_cob_2=outcome2.ending_cars)
                cars_rented = outcome1.cars_rented + outcome2.cars_rented
                probability = probability1 * probability2
                outcome: tuple = (new_state, cars_rented)
                outcome_dict[outcome] += probability
                # probability: Optional[float] = outcome_dict.get(outcome)
                # if probability:
                #     outcome_dict[outcome] += probability
                # else:
                #     outcome_dict[outcome] = probability

        outcome_list: list[Outcome] = []
        for outcome, probability in outcome_dict.items():
            new_state: State = outcome[0]
            cars_rented: int = outcome[1]
            revenue: float = cars_rented * self._rental_revenue
            reward: float = revenue - total_costs
            outcome_list.append(Outcome(new_state, reward, probability))

        return outcome_list

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        total_costs: float = self._calc_cost_of_transfers(action.transfer_1_to_2)
        if self._extra_rules:
            total_costs += self._location_1.parking_costs(state.cars_cob_1)
            total_costs += self._location_2.parking_costs(state.cars_cob_2)

        cars_sob_1 = state.cars_cob_1 - action.transfer_1_to_2
        cars_sob_2 = state.cars_cob_2 + action.transfer_1_to_2

        outcome1: LocationOutcome = self._location_1.draw_outcome(cars_sob_1)
        outcome2: LocationOutcome = self._location_2.draw_outcome(cars_sob_2)

        new_state = State(is_terminal=False, cars_cob_1=outcome1.ending_cars, cars_cob_2=outcome2.ending_cars)
        cars_rented = outcome1.cars_rented + outcome2.cars_rented
        revenue: float = cars_rented * self._rental_revenue
        reward: float = revenue - total_costs
        return Response(reward, new_state)

    def _calc_cost_of_transfers(self, transfer_1_to_2: int) -> float:
        """This will change for second part of problem"""
        if self._extra_rules and transfer_1_to_2 >= 1:
            transfers_incurring_cost = transfer_1_to_2 - 1  # one free ride
        else:
            transfers_incurring_cost = abs(transfer_1_to_2)
        transfer_cost = self._transfer_cost * transfers_incurring_cost
        return transfer_cost
