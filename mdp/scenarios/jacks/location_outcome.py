from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LocationOutcome:
    cars_rented: int
    ending_cars: int
    probability: float
