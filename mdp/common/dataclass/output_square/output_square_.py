from __future__ import annotations
from typing import Optional
import dataclasses

from mdp.common.dataclass import xy
from mdp.common.dataclass.output_square import move_value


@dataclasses.dataclass
class OutputSquare:
    v_value: Optional[float] = None
    policy_value: Optional[float] = None
    move_values: dict[xy.XY, move_value.MoveValue] = dataclasses.field(default_factory=dict)
