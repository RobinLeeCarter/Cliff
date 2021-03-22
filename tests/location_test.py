from __future__ import annotations
# from typing import TYPE_CHECKING

from scipy import stats

from mdp.common import Distribution

# _car_count: list[int] = []
# _demand_distribution: Distribution[int] = Distribution()
_max_cars = 5


def _poisson(lambda_: float, n: int) -> float:
    return stats.poisson.pmf(k=n, mu=lambda_)


_car_count = [c for c in range(_max_cars + 1)]
print(_car_count)

_demand_distribution = Distribution({c: _poisson(4.0, c) for c in range(_max_cars + 1)})
print(_demand_distribution)

_demand_distribution[_max_cars] += 1.0 - sum(_demand_distribution.values())
print(_demand_distribution)

_demand_distribution.self_check()
