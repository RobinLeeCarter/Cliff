from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Tally:
    count: int
    average: float
