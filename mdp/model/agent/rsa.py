from __future__ import annotations
from typing import Optional
from dataclasses import dataclass


@dataclass
class RSA:
    r: Optional[float]
    s: Optional[int]
    a: Optional[int]

    @property
    def tuple(self) -> tuple:
        return self.r, self.s, self.a
