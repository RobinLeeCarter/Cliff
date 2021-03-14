from typing import Optional

import numpy as np
from scipy import stats

from mdp.scenarios.jacks.dynamics.dict_zero import DictZero
from mdp.scenarios.jacks.dynamics.location_outcome import LocationOutcome
# from mdp.scenarios.jacks.dynamics.location_outcomes import LocationOutcomes


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
        # could have used dictionary of dictionaries but fast enough and that could be confusing
        self.outcome_lookup: dict[int, list[LocationOutcome]] = {}
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
        # self._daily_outcome_tables()
        # print(f"daily_outcomes = {self._counter}")
        # exit()

    def _rental_return_prob(self):
        car_count = [c for c in range(self._max_cars + 1)]
        self._demand_prob = np.array([self._poisson(self._rental_rate, c) for c in car_count])
        self._return_prob = np.array([self._poisson(self._return_rate, c) for c in car_count])
        self._demand_prob[-1] += 1.0 - np.sum(self._demand_prob)
        self._return_prob[-1] += 1.0 - np.sum(self._return_prob)
        # print(self._demand_prob)
        # print(self._return_prob)

    def _poisson(self, lambda_: float, n: int):
        return stats.poisson.pmf(k=n, mu=lambda_)

    def get_outcomes(self, starting_cars: int) -> dict[LocationOutcome, float]:
        location_outcomes: DictZero[LocationOutcome, float] = DictZero()
        for car_demand, demand_probability in enumerate(self._demand_prob):
            cars_rented = min(starting_cars, car_demand)
            for cars_returned, return_probability in enumerate(self._return_prob):
                ending_cars = starting_cars - cars_rented + cars_returned
                if ending_cars > self._max_cars:
                    ending_cars = self._max_cars

                probability = demand_probability * return_probability

                if probability > 0.0:
                    location_outcome = LocationOutcome(ending_cars, cars_rented)
                    location_outcomes[location_outcome] += probability
                    # outcome_: Optional[LocationOutcome] = None
                    # for location_outcome_ in outcome_list:
                    #     if location_outcome_.ending_cars == ending_cars and \
                    #             location_outcome_.cars_rented == cars_rented:
                    #         outcome_ = location_outcome_
                    #         outcome_.probability += probability
                    #         break
                    # if not outcome_:
                    #     outcome = LocationOutcome(
                    #         ending_cars=ending_cars,
                    #         cars_rented=cars_rented,
                    #         probability=probability,
                    #     )
                    #     outcome_list.append(outcome)
                    #     # self._counter += 1
        return location_outcomes

    def _daily_outcome_tables(self):
        for starting_cars in range(self._max_cars + 1):
            outcome_list: list[LocationOutcome] = []
            self.outcome_lookup[starting_cars] = outcome_list
            for car_demand, demand_probability in enumerate(self._demand_prob):
                cars_rented = min(starting_cars, car_demand)
                for cars_returned, return_probability in enumerate(self._return_prob):
                    ending_cars = starting_cars - cars_rented + cars_returned
                    if ending_cars > self._max_cars:
                        ending_cars = self._max_cars

                    # part of p(ending_cars, cars_rented | starting_cars)
                    # aka eg. p(s1', r1 | s1, a)
                    probability = demand_probability * return_probability
                    # part of p(ending_cars, cars_rented | starting_cars) x cars_rented
                    # aka eg. p(s1', r1 | s1, a) x r1
                    probability_x_cars_rented = probability * cars_rented

                    # summarise by ending_cars

                    # find outcome to append to or add to list
                    outcome_: Optional[LocationOutcome] = None
                    for location_outcome_ in outcome_list:
                        if location_outcome_.ending_cars == ending_cars:
                            outcome_ = location_outcome_
                            outcome_.probability += probability
                            # outcome_.probability_x_cars_rented += probability_x_cars_rented
                            break
                    if not outcome_:
                        outcome = LocationOutcome(
                            ending_cars=ending_cars,
                            probability=probability,
                            # probability_x_cars_rented=probability_x_cars_rented
                        )
                        outcome_list.append(outcome)
                        self._counter += 1

    # def day_distribution(self, starting_cars) -> Generator[LocationOutcome, None, None]:
    #     for car_demand, demand_probability in enumerate(self._demand_prob):
    #         cars_rented = min(starting_cars, car_demand)
    #         for cars_returned, return_probability in enumerate(self._return_prob):
    #             joint_probability = demand_probability * return_probability
    #             ending_cars = starting_cars - cars_rented + cars_returned
    #             if ending_cars > 20:
    #                 ending_cars = 20
    #             yield LocationOutcome(cars_rented, ending_cars, joint_probability)

    def parking_costs(self, end_cars: int) -> float:
        if end_cars > 10:
            return self._excess_parking_cost
        else:
            return 0.0
