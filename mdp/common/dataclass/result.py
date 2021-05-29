from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import dataclasses

import numpy as np

if TYPE_CHECKING:
    from mdp.model.breakdown.recorder import Recorder
    from mdp.model.algorithm.value_function.state_function import StateFunction
    from mdp.model.algorithm.value_function.state_action_function import StateActionFunction


@dataclasses.dataclass
class Result:
    algorithm_title: str
    recorder: Optional[Recorder] = None
    policy_vector: Optional[np.ndarray] = None
    V: Optional[StateFunction] = None
    Q: Optional[StateActionFunction] = None
