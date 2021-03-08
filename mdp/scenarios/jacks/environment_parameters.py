from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp import common


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    max_cars: Optional[int] = None
    max_transfers: Optional[int] = None
    rental_rate_1: Optional[float] = None
    rental_rate_2: Optional[float] = None
    return_rate_1: Optional[float] = None
    return_rate_2: Optional[float] = None
    rental_revenue: Optional[float] = None
    transfer_cost: Optional[float] = None


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.JACKS,
    max_cars=20,
    max_transfers=5,
    rental_rate_1=3.0,
    rental_rate_2=4.0,
    return_rate_1=3.0,
    return_rate_2=2.0,
    rental_revenue=10.0,
    transfer_cost=2.0,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
