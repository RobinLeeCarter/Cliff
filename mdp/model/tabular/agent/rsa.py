from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RSA:
    r: float
    s: int
    a: int

    @property
    def tuple(self) -> tuple[float, int, int]:
        return self.r, self.s, self.a
