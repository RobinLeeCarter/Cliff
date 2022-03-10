from __future__ import annotations
from typing import Optional
import dataclasses
import copy

import numpy as np

from mdp import common
from mdp.scenario.windy.model import grids


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    upward_wind: Optional[np.ndarray] = None
    random_wind: Optional[bool] = None
    random_wind_choices: Optional[np.ndarray] = None


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.WINDY,
    actions_list=common.ActionsList.FOUR_MOVES,
    grid=grids.GRID_1,
    verbose=False,
    upward_wind=grids.UPWARD_WIND_1,
    random_wind=False,
    random_wind_choices=grids.RANDOM_WIND_CHOICES_1
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
