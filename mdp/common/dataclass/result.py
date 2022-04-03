from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from mdp.model.breakdown.recorder import Recorder


@dataclass
class Result:
    recorder: Optional[Recorder] = None
    policy_vector: Optional[np.ndarray] = None
    v_vector: Optional[np.ndarray] = None
    q_matrix: Optional[np.ndarray] = None
    cum_timestep: Optional[int] = None
