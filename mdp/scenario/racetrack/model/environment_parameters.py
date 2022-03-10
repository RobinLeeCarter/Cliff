from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp import common

from mdp.scenario.racetrack.model import grids


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    min_velocity: Optional[int] = None
    max_velocity: Optional[int] = None
    min_acceleration: Optional[int] = None
    max_acceleration: Optional[int] = None
    extra_reward_for_failure: Optional[float] = None
    skid_probability: Optional[float] = None


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.RACETRACK,
    verbose=False,
    grid=grids.TRACK_1,
    min_velocity=0,
    max_velocity=4,
    min_acceleration=-1,
    max_acceleration=+1,
    extra_reward_for_failure=0.0,
    skid_probability=0.1,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
