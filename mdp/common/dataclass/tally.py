from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class Tally:
    count: int
    average: float
