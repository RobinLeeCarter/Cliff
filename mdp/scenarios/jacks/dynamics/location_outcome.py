from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LocationOutcome:
    ending_cars: int                    # unique key
    cars_rented: int                    # unique key
    # probability: float                  # p(s1'|s,a) == sum_over_r1( p(s1',r1|s,a) )
