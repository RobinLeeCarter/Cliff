from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import dataclasses

import numpy as np

if TYPE_CHECKING:
    from mdp.model.breakdown.recorder import Recorder


@dataclasses.dataclass
class Result:
    algorithm_title: Optional[str] = None
    recorder: Optional[Recorder] = None
    policy_vector: Optional[np.ndarray] = None
    v_vector: Optional[np.ndarray] = None
    q_matrix: Optional[np.ndarray] = None
    cum_timestep: Optional[int] = None
