from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.scenario.racetrack.model import grids
from mdp.model.tabular.environment.tabular_environment_parameters import TabularEnvironmentParameters


@dataclass
class EnvironmentParameters(TabularEnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.RACETRACK
    grid: grids = grids.TRACK_1
    min_velocity: int = 0
    max_velocity: int = 4
    min_acceleration: int = -1
    max_acceleration: int = +1
    extra_reward_for_failure: float = 0.0
    skid_probability: float = 0.1
