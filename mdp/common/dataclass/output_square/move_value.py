from __future__ import annotations
import dataclasses
from typing import Optional

from mdp.common.dataclass.xy import XY


@dataclasses.dataclass
class MoveValue:
    move: XY
    q_value: Optional[float] = None
    is_policy: bool = False
