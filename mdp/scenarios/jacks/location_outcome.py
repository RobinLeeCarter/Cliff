from __future__ import annotations
from dataclasses import dataclass


@dataclass()
class LocationOutcome:
    ending_cars: int                    # unique key
    probability: float                  # p(s1'|s,a) == sum_over_r1( p(s1',r1|s,a) )
    probability_x_cars_rented: float    # sum_over_r1( p(s1',r1|s1,a) . r1 )
