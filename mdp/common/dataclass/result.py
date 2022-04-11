from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    from mdp.model.breakdown.recorder import Recorder
    from mdp.model.non_tabular.agent.reward_state_action import Trajectory


@dataclass
class Result:
    recorder: Optional[Recorder] = None
    policy_vector: Optional[np.ndarray] = None
    v_vector: Optional[np.ndarray] = None
    q_matrix: Optional[np.ndarray] = None
    cum_timestep: Optional[int] = None
    delta_w_vector: Optional[np.ndarray] = None
    trajectories: list[Trajectory] = field(default_factory=list)    # non-tabular
