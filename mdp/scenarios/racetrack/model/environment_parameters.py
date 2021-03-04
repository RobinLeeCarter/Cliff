from __future__ import annotations
from typing import Optional
import dataclasses

import numpy as np

from mdp import common
from mdp.common.dataclass import environment_parameters
from mdp.scenarios.racetrack.model import tracks


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    track: Optional[np.ndarray] = None
    min_velocity: Optional[int] = None
    max_velocity: Optional[int] = None
    min_acceleration: Optional[int] = None
    max_acceleration: Optional[int] = None
    extra_reward_for_failure: Optional[float] = None
    skid_probability: Optional[float] = None


# TODO: how to instantiate with this as default in scenarios rather than common

default: EnvironmentParameters = EnvironmentParameters(
    verbose=environment_parameters.default.verbose,
    track=tracks.TRACK_1,
    min_velocity=0,
    max_velocity=4,
    min_acceleration=-1,
    max_acceleration=+1,
    extra_reward_for_failure=-40.0,
    skid_probability=0.1,
)
