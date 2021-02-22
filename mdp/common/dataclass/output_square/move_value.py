from __future__ import annotations
import dataclasses

from mdp.common import named_tuples


@dataclasses.dataclass
class MoveValue:
    move: named_tuples.XY
    q_value: float
    is_policy: bool = False
