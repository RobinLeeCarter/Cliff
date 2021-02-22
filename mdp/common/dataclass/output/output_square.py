from __future__ import annotations
from typing import Optional
import dataclasses

import numpy as np

from mdp.common.dataclass.output import move_value


@dataclasses.dataclass
class OutputSquare:
    v_value: Optional[float] = None
    q_values: np.ndarray = np.array([], dtype=move_value.MoveValue)
