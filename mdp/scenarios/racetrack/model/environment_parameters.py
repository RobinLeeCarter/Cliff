from __future__ import annotations
from typing import Optional
import dataclasses
import copy

import numpy as np

from mdp import common


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    track: Optional[np.ndarray] = None
    min_velocity: Optional[int] = None
    max_velocity: Optional[int] = None
    min_acceleration: Optional[int] = None
    max_acceleration: Optional[int] = None
    extra_reward_for_failure: Optional[float] = None
    skid_probability: Optional[float] = None


default: EnvironmentParameters = EnvironmentParameters(
    actions_list=common.ActionsList.FOUR_MOVES,
    random_wind=False,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
