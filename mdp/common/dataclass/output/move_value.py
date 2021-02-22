from __future__ import annotations
# from typing import Optional
import dataclasses

from mdp.common import named_tuples


@dataclasses.dataclass
class MoveValue:
    move: named_tuples.XY
    value: float
    is_policy: bool = False
