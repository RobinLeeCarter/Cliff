from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.common.dataclass.xy import XY


@dataclass
class MoveValue:
    move: XY
    q_value: Optional[float] = None
    is_policy: bool = False
