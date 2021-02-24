from __future__ import annotations
import dataclasses
from typing import Optional

from mdp.common import named_tuples


@dataclasses.dataclass
class MoveValue:
    move: named_tuples.XY
    q_value: Optional[float] = None
    is_policy: bool = False
