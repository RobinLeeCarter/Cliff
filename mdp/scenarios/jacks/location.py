from typing import Generator

import numpy as np
from scipy import stats

from mdp.scenarios.jacks import location_outcome


class Location:
    def __init__(self, max_cars: int, rental_rate: float, return_rate: float, excess_parking_cost: float):
        self._max_cars: int = max_cars
        self._rental_rate: float = rental_rate
        self._return_rate: float = return_rate
        self._excess_parking_cost: float = excess_parking_cost

        # self._revenue_per_car: float = 10.0
        # self._park_penalty: float = 4.0

        self._demand_prob: np.ndarray = np.zeros(self._max_cars + 1, float)
        self._return_prob: np.ndarray = np.zeros(self._max_cars + 1, float)

        # for each starting_cars find possible outcomes
        self.daily_outcomes: dict[int, list[location_outcome.LocationOutcome]] = {}
        self._counter: int = 0

        # given starting_cars as input, value is expected revenue
        # E[r[l] | s, a]
        # self._expected_revenue: np.ndarray = np.zeros(self._max_cars + 1, float)

        # given starting_cars as first value, value is probability of ending_cars
        # Pr(s'[l] | s, a)
        # self._prob_ending_cars: np.ndarray = np.zeros(shape=(self._max_cars + 1, self._max_cars + 1), dtype=float)

        self._build()

    def _build(self):
        self._rental_return_prob()
        self._daily_outcome_tables()
        print(f"daily_outcomes = {self._counter}")
        # exit()

    def _rental_return_prob(self):
        car_count = [c for c in range(self._max_cars + 1)]
        self._demand_prob = np.array([self._poisson(self._rental_rate, c) for c in car_count])
        self._return_prob = np.array([self._poisson(self._return_rate, c) for c in car_count])
        self._demand_prob[-1] += 1.0 - np.sum(self._demand_prob)
        self._return_prob[-1] += 1.0 - np.sum(self._return_prob)

    def _poisson(self, lambda_: float, n: int):
        return stats.poisson.pmf(k=n, mu=lambda_)

    def _daily_outcome_tables(self):
        for starting_cars in range(self._max_cars + 1):
            outcome_list: list[location_outcome.LocationOutcome] = []
            self.daily_outcomes[starting_cars] = outcome_list
            for car_demand, demand_probability in enumerate(self._demand_prob):
                cars_rented = min(starting_cars, car_demand)
                for cars_returned, return_probability in enumerate(self._return_prob):
                    joint_probability = demand_probability * return_probability
                    ending_cars = starting_cars - cars_rented + cars_returned
                    if ending_cars > 20:
                        ending_cars = 20

                    added: bool = False
                    for location_outcome_ in outcome_list:
                        if location_outcome_.ending_cars == ending_cars and \
                                location_outcome_.cars_rented == cars_rented:
                            location_outcome_.probability += joint_probability
                            added = True
                            break
                    if not added:
                        outcome = location_outcome.LocationOutcome(ending_cars, cars_rented, joint_probability)
                        self.daily_outcomes[starting_cars].append(outcome)
                        self._counter += 1

    def day_distribution(self, starting_cars) -> Generator[location_outcome.LocationOutcome, None, None]:
        for car_demand, demand_probability in enumerate(self._demand_prob):
            cars_rented = min(starting_cars, car_demand)
            for cars_returned, return_probability in enumerate(self._return_prob):
                joint_probability = demand_probability * return_probability
                ending_cars = starting_cars - cars_rented + cars_returned
                if ending_cars > 20:
                    ending_cars = 20
                yield location_outcome.LocationOutcome(cars_rented, ending_cars, joint_probability)

    def parking_costs(self, end_cars: int) -> float:
        if end_cars > 10:
            return self._excess_parking_cost
        else:
            return 0.0
