from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class RGBA:
    r: int
    g: int
    b: int
    a: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return dataclasses.astuple(self)
