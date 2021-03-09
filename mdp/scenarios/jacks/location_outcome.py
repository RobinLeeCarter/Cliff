from __future__ import annotations
from dataclasses import dataclass


@dataclass()
class LocationOutcome:
    ending_cars: int
    cars_rented: int
    probability: float
