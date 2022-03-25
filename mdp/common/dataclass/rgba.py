from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RGBA:
    r: int
    g: int
    b: int
    a: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.r, self.g, self.b, self.a
