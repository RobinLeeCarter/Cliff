from scipy import stats

from mdp.common import DictZero, Distribution
from mdp.scenarios.jacks.model.dynamics.location_outcome import LocationOutcome


class Location:
    def __init__(self, max_cars: int, rental_rate: float, return_rate: float, excess_parking_cost: float):
        self._max_cars: int = max_cars
        self._rental_rate: float = rental_rate
        self._return_rate: float = return_rate
        self._excess_parking_cost: float = excess_parking_cost

        self._car_count: list[int] = []
        self._demand_distribution: Distribution[int] = Distribution()
        self._return_distribution: Distribution[int] = Distribution()

        # for each starting_cars find possible outcomes
        # dict[starting_cars, dict[LocationOutcome, probability]]
        self.outcome_distributions: dict[int, Distribution[LocationOutcome]] = {}
        self._counter: int = 0

        # summaries
        # dict[starting_cars, cars_rented * probability]
        self.expected_cars_rented: dict[int, float] = {}
        # dict[starting_cars, dict[ending_cars, probability]]
        self.ending_cars_distribution: dict[int, Distribution[int]] = {}
        # dict[starting_cars, dict[ending_cars, cars_rented_x_probability]]
        self.cars_rented_x_probability_by_ending_cars: dict[int, dict[int, float]] = {}
        # dict[starting_cars, dict[ending_cars, expected_cars_rented]]
        self.expected_cars_rented_by_ending_cars: dict[int, dict[int, float]] = {}

        # given starting_cars as input, value is expected revenue
        # E[r[l] | s, a]
        # self._expected_revenue: np.ndarray = np.zeros(self._max_cars + 1, float)

        # given starting_cars as first value, value is probability of ending_cars
        # Pr(s'[l] | s, a)
        # self._prob_ending_cars: np.ndarray = np.zeros(shape=(self._max_cars + 1, self._max_cars + 1), dtype=float)

    def build(self):
        self._build_rental_return_distributions()
        self._build_outcome_distributions()
        for distribution in self.outcome_distributions.values():
            for _ in distribution.keys():
                self._counter += 1
        # print(f"daily_outcomes = {self._counter}")
        self._build_summaries()

    def _build_rental_return_distributions(self):
        self._car_count = [c for c in range(self._max_cars + 1)]

        self._demand_distribution = Distribution({c: self._poisson(self._rental_rate, c)
                                                  for c in range(self._max_cars + 1)})
        self._demand_distribution[self._max_cars] += 1.0 - sum(self._demand_distribution.values())
        self._demand_distribution.seal()

        self._return_distribution = Distribution({c: self._poisson(self._return_rate, c)
                                                  for c in range(self._max_cars + 1)})
        self._return_distribution[self._max_cars] += 1.0 - sum(self._return_distribution.values())
        self._return_distribution.seal()

    def _poisson(self, lambda_: float, n: int) -> float:
        return stats.poisson.pmf(k=n, mu=lambda_)

    def _build_outcome_distributions(self):
        for starting_cars in self._car_count:
            self._build_outcome_distribution(starting_cars)

    def _build_outcome_distribution(self, starting_cars: int):
        outcome_distribution: Distribution[LocationOutcome, float] = Distribution()
        # cars_rented_x_probability: float = 0.0

        for car_demand, demand_probability in self._demand_distribution.items():
            cars_rented = self._get_cars_rented(starting_cars, car_demand)
            for cars_returned, return_probability in self._return_distribution.items():
                ending_cars = self._get_ending_cars(starting_cars, cars_rented, cars_returned)
                probability = demand_probability * return_probability

                if probability > 0.0:
                    location_outcome = LocationOutcome(ending_cars, cars_rented)
                    outcome_distribution[location_outcome] += probability
        outcome_distribution.seal()

        self.outcome_distributions[starting_cars] = outcome_distribution

    def _get_cars_rented(self, starting_cars: int, car_demand: int) -> int:
        return min(starting_cars, car_demand)

    def _get_ending_cars(self, starting_cars: int, cars_rented: int, cars_returned: int) -> int:
        ending_cars = starting_cars - cars_rented + cars_returned
        if ending_cars > self._max_cars:
            ending_cars = self._max_cars
        return ending_cars

    def _build_summaries(self):
        for starting_cars in self.outcome_distributions.keys():
            expected_cars_rented: float = 0.0
            ending_cars_distribution: Distribution[int] = Distribution()
            cars_rented_x_probability_by_ending_cars: DictZero[int, float] = DictZero()

            for outcome, probability in self.outcome_distributions[starting_cars].items():
                cars_rented_x_probability = outcome.cars_rented * probability

                expected_cars_rented += cars_rented_x_probability
                ending_cars_distribution[outcome.ending_cars] += probability
                cars_rented_x_probability_by_ending_cars[outcome.ending_cars] += cars_rented_x_probability
            ending_cars_distribution.seal()

            expected_cars_rented_by_ending_cars: DictZero[int, float] = DictZero()
            for ending_cars, ending_cars_probability in ending_cars_distribution.items():
                cars_rented_x_probability = cars_rented_x_probability_by_ending_cars[ending_cars]
                # E[r|s,a,s'] = Sum_over_r( p(r,s'|s,a).r ) / p(s'|s,a)
                conditional_expected_cars_rented = cars_rented_x_probability / ending_cars_probability
                expected_cars_rented_by_ending_cars[ending_cars] = conditional_expected_cars_rented

            self.expected_cars_rented[starting_cars] = expected_cars_rented
            self.ending_cars_distribution[starting_cars] = ending_cars_distribution
            self.cars_rented_x_probability_by_ending_cars[starting_cars] = cars_rented_x_probability_by_ending_cars
            self.expected_cars_rented_by_ending_cars[starting_cars] = expected_cars_rented_by_ending_cars

    def get_outcome_distribution(self, starting_cars: int) -> Distribution[LocationOutcome]:
        return self.outcome_distributions[starting_cars]

    def get_ending_cars_distribution(self, starting_cars: int) -> Distribution[int]:
        return self.ending_cars_distribution[starting_cars]

    def get_transition_probability(self, starting_cars: int, ending_cars: int) -> float:
        return self.ending_cars_distribution[starting_cars][ending_cars]

    def get_expected_cars_rented_given_ending_cars(self, starting_cars: int, ending_cars: int) -> float:
        return self.expected_cars_rented_by_ending_cars[starting_cars][ending_cars]

    def draw_outcome(self, starting_cars: int) -> LocationOutcome:
        return self.outcome_distributions[starting_cars].draw_one()
        # outcome_distribution = self.outcome_distributions[starting_cars]
        # outcome: LocationOutcome = random.choices(
        #     population=list(outcome_distribution.keys()),
        #     weights=list(outcome_distribution.values())
        # )[0]
        # return outcome
        # car_demand: int = random.choices(population=self._car_count, weights=self._demand_prob)[0]
        # cars_rented = self._get_cars_rented(starting_cars, car_demand)
        # cars_returned = random.choices(population=self._car_count, weights=self._return_prob)[0]
        # ending_cars = self._get_ending_cars(starting_cars, cars_rented, cars_returned)
        # return LocationOutcome(ending_cars, cars_rented)

    def parking_costs(self, end_cars: int) -> float:
        if end_cars > 10:
            return self._excess_parking_cost
        else:
            return 0.0
