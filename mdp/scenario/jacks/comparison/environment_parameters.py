from __future__ import annotations
from dataclasses import dataclass

from mdp import common


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.JACKS
    max_cars: int = 20
    max_transfers: int = 5
    rental_rate_1: float = 3.0
    return_rate_1: float = 3.0
    rental_rate_2: float = 4.0
    return_rate_2: float = 2.0
    rental_revenue: float = 10.0
    transfer_cost: float = 2.0

    extra_rules: bool = False
    excess_parking_cost: float = 4.0
