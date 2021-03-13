from __future__ import annotations
import dataclasses
from typing import Optional

from mdp import common


@dataclasses.dataclass
class MoveValue:
    move: common.XY
    q_value: Optional[float] = None
    is_policy: bool = False
