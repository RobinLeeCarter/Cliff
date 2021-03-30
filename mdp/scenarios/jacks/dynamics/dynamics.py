from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.jacks.model.action import Action
    from mdp.scenarios.jacks.model.environment import Environment
    from mdp.scenarios.jacks.model.environment_parameters import EnvironmentParameters
    from mdp.scenarios.jacks.dynamics.location_outcome import LocationOutcome

import random

from mdp.common import Distribution
from mdp.model.environment import dynamics

from mdp.scenarios.jacks.model.state import State
from mdp.scenarios.jacks.model.response import Response
from mdp.scenarios.jacks.dynamics.location import Location


class Dynamics(dynamics.Dynamics):
    def __init__(self, environment_: Environment, environment_parameters: EnvironmentParameters):
        super().__init__(environment_, environment_parameters)

        # downcast
        self._environment: Environment = self._environment

        self._max_cars: int = environment_parameters.max_cars
        self._rental_revenue: float = environment_parameters.rental_revenue
        self._transfer_cost: float = environment_parameters.transfer_cost
        self._extra_rules: bool = environment_parameters.extra_rules

        self._location_1: Location = Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters.rental_rate_1,
            return_rate=environment_parameters.return_rate_1,
            excess_parking_cost=environment_parameters.excess_parking_cost,
        )
        self._location_2: Location = Location(
            max_cars=self._max_cars,
            rental_rate=environment_parameters.rental_rate_2,
            return_rate=environment_parameters.return_rate_2,
            excess_parking_cost=environment_parameters.excess_parking_cost,
        )

        # reused "current" variables
        self._total_costs: float = 0.0
        self._starting_cars_1: int = 0
        self._starting_cars_2: int = 0

        # summaries
        self._expected_reward: dict[tuple[State, Action], float] = {}
        self._next_state_distribution: dict[tuple[State, Action], Distribution[State]] = {}

    def build(self):
        """
        key functions to build summaries for are:
        - get_expected_reward
        - get_next_state_distribution
        """
        self._location_1.build()
        self._location_2.build()
        self._build_expected_reward_summary()
        self._build_next_state_distribution_summary()
        super().build()

    def _build_expected_reward_summary(self):
        for state in self._environment.states:
            for action in self._environment.actions_for_state(state):
                self._expected_reward[(state, action)] = self._calc_expected_reward(state, action)

    def _build_next_state_distribution_summary(self):
        for state in self._environment.states:
            for action in self._environment.actions_for_state(state):
                self._next_state_distribution[(state, action)] = self._calc_next_state_distribution(state, action)

    def _calc_expected_reward(self, state: State, action: Action) -> float:
        """
        r(s,a) = E[Rt | S(t-1)=s, A(t-1)=a] = Sum_over_s'_r( p(s',r|s,a).r )
        expected reward for a (state, action)
        """
        self._calc_start_of_day(state, action)
        expected_cars_rented_1 = self._location_1.expected_cars_rented[self._starting_cars_1]
        expected_cars_rented_2 = self._location_2.expected_cars_rented[self._starting_cars_2]
        expected_cars_rented = expected_cars_rented_1 + expected_cars_rented_2
        expected_reward = self._calc_reward(expected_cars_rented)
        return expected_reward

    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        r(s,a) = E[Rt | S(t-1)=s, A(t-1)=a] = Sum_over_s'_r( p(s',r|s,a).r )
        expected reward for a (state, action)
        """
        return self._expected_reward[(state, action)]

    def get_probability_x_reward(self, state: State, action: Action, next_state: State) -> float:
        """
        Sum_over_r( p(s',r|s,a).r )
        probability_x_reward for a (state, action) given the next state
        """
        self._calc_start_of_day(state, action)
        cars_rented_x_probability1 = self._location_1.get_expected_cars_rented_given_ending_cars(
            self._starting_cars_1, next_state.ending_cars_1)
        cars_rented_x_probability2 = self._location_2.get_expected_cars_rented_given_ending_cars(
            self._starting_cars_2, next_state.ending_cars_2)
        cars_rented_x_probability = cars_rented_x_probability1 + cars_rented_x_probability2

        probability1 = self._location_1.get_transition_probability(self._starting_cars_1, next_state.ending_cars_1)
        probability2 = self._location_2.get_transition_probability(self._starting_cars_2, next_state.ending_cars_2)
        probability = probability1 * probability2

        # **** THIS MAY BE WRONG ****
        probability_x_reward = self._calc_reward(cars_rented_x_probability, probability)
        return probability_x_reward

    def get_state_transition_probability(self, state: State, action: Action, next_state: State) -> float:
        """
        p(s'|s,a) = Sum_over_r( p(s',r|s,a) )
        probability of a next state for a (state, action)
        """
        self._calc_start_of_day(state, action)
        probability1 = self._location_1.get_transition_probability(self._starting_cars_1, next_state.ending_cars_1)
        probability2 = self._location_2.get_transition_probability(self._starting_cars_2, next_state.ending_cars_2)
        probability = probability1 * probability2
        return probability

    def _calc_next_state_distribution(self, state: State, action: Action) -> Distribution[State]:
        """
        dict[ s', p(s'|s,a) ]
        distribution of next states for a (state, action)
        """
        self._calc_start_of_day(state, action)
        ending_cars_distribution1 = self._location_1.get_ending_cars_distribution(self._starting_cars_1)
        ending_cars_distribution2 = self._location_2.get_ending_cars_distribution(self._starting_cars_2)

        next_state_distribution: Distribution[State] = Distribution()
        for ending_cars1, probability1 in ending_cars_distribution1.items():
            for ending_cars2, probability2 in ending_cars_distribution2.items():
                next_state = State(is_terminal=False, ending_cars_1=ending_cars1, ending_cars_2=ending_cars2)
                probability = probability1 * probability2
                next_state_distribution[next_state] = probability
        next_state_distribution.self_check()
        return next_state_distribution

    def get_state_transition_distribution(self, state: State, action: Action) -> Distribution[State]:
        """
        dict[ s', p(s'|s,a) ]
        distribution of next states for a (state, action)
        """
        return self._next_state_distribution[(state, action)]

    def get_summary_outcomes(self, state: State, action: Action) -> Distribution[Response]:
        """
        dict of possible responses for a single state and action
        with the expected_reward given in place of reward
        """
        self._calc_start_of_day(state, action)
        l1 = self._location_1
        l2 = self._location_2

        ending_cars_dist1: dict[int, float] = l1.get_ending_cars_distribution(self._starting_cars_1)
        ending_cars_dist2: dict[int, float] = l2.get_ending_cars_distribution(self._starting_cars_2)

        response_distribution: Distribution[Response] = Distribution()
        for ending_cars1, probability1 in ending_cars_dist1.items():
            cars_rented1 = l1.get_expected_cars_rented_given_ending_cars(self._starting_cars_1, ending_cars1)
            for ending_cars2, probability2 in ending_cars_dist2.items():
                cars_rented2 = l2.get_expected_cars_rented_given_ending_cars(self._starting_cars_2, ending_cars2)
                cars_rented = cars_rented1 + cars_rented2
                new_state = State(is_terminal=False, ending_cars_1=ending_cars1, ending_cars_2=ending_cars2)
                probability = probability1 * probability2
                reward = self._calc_reward(cars_rented)
                response = Response(reward, new_state)
                response_distribution[response] += probability
        response_distribution.self_check()
        return response_distribution

    def get_all_outcomes(self, state: State, action: Action) -> Distribution[Response]:
        """
        dict of possible responses for a single state and action
        could be used for one state, action in theory
        but too many for all states and actions so potentially not useful in practice
        """
        self._calc_start_of_day(state, action)
        l1 = self._location_1
        l2 = self._location_2

        outcomes1: Distribution[LocationOutcome] = l1.get_outcome_distribution(self._starting_cars_1)
        outcomes2: Distribution[LocationOutcome] = l2.get_outcome_distribution(self._starting_cars_2)

        # collate (s', r)
        # outcome_dict: dict[(next_state, reward), probability]
        # outcome_dict: DictZero[tuple[State, float], float] = DictZero()
        response_distribution: Distribution[Response] = Distribution()
        for outcome1, probability1 in outcomes1.items():
            for outcome2, probability2 in outcomes2.items():
                cars_rented = outcome1.cars_rented + outcome2.cars_rented
                new_state = State(is_terminal=False,
                                  ending_cars_1=outcome1.ending_cars,
                                  ending_cars_2=outcome2.ending_cars)
                probability = probability1 * probability2
                reward = self._calc_reward(cars_rented)
                response = Response(reward, new_state)
                response_distribution[response] += probability
        response_distribution.self_check()
        return response_distribution

    def get_a_start_state(self) -> State:
        return random.choice(self._environment.states)

    def draw_response(self, state: State, action: Action) -> Response:
        """
        draw a single outcome for a single state and action
        standard call for episodic algorithms
        """
        self._calc_start_of_day(state, action)
        outcome1: LocationOutcome = self._location_1.draw_outcome(self._starting_cars_1)
        outcome2: LocationOutcome = self._location_2.draw_outcome(self._starting_cars_2)
        new_state = State(is_terminal=False, ending_cars_1=outcome1.ending_cars, ending_cars_2=outcome2.ending_cars)
        cars_rented = outcome1.cars_rented + outcome2.cars_rented
        reward: float = self._calc_reward(cars_rented)
        return Response(reward, new_state)

    def _calc_start_of_day(self, state: State, action: Action):
        self._total_costs: float = self._calc_cost_of_transfers(action.transfer_1_to_2)
        if self._extra_rules:
            self._total_costs += self._location_1.parking_costs(state.ending_cars_1)
            self._total_costs += self._location_2.parking_costs(state.ending_cars_2)

        self._starting_cars_1 = state.ending_cars_1 - action.transfer_1_to_2
        self._starting_cars_2 = state.ending_cars_2 + action.transfer_1_to_2

    def _calc_reward(self, cars_rented: float, partial_probability: float = 1.0) -> float:
        """caters for case where calculating probability * reward for only part of the probability distribution"""
        revenue: float = cars_rented * self._rental_revenue
        reward: float = revenue - (self._total_costs * partial_probability)
        return reward

    def _calc_cost_of_transfers(self, transfer_1_to_2: int) -> float:
        """This will change for second part of problem"""
        if self._extra_rules and transfer_1_to_2 >= 1:
            transfers_incurring_cost = transfer_1_to_2 - 1  # one free ride
        else:
            transfers_incurring_cost = abs(transfer_1_to_2)
        transfer_cost = self._transfer_cost * transfers_incurring_cost
        return transfer_cost
