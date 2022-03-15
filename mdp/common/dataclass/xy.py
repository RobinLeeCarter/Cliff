from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class XY:
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y
