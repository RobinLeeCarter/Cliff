from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from mdp.common.dataclass.xy import XY
from mdp.common.dataclass.output_square.move_value import MoveValue


@dataclass
class OutputSquare:
    v_value: Optional[float] = None
    policy_value: Optional[float] = None
    move_values: dict[XY, MoveValue] = field(default_factory=dict)
