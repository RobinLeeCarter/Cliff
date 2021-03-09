from __future__ import annotations
from dataclasses import dataclass


@dataclass()
class LocationSummary:
    sum_cars_rented_x_probability: float
    probability: float
