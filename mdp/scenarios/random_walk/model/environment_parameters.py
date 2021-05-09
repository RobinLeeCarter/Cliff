from __future__ import annotations
from typing import Optional
import dataclasses
import copy

import numpy as np

from mdp import common

from mdp.scenarios.random_walk.model import grids


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    v_optimal: Optional[np.ndarray] = None
    random_move_choices: Optional[np.ndarray] = None


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.ScenarioType.RANDOM_WALK,
    actions_list=common.ActionsList.NO_ACTIONS,
    grid=grids.GRID,
    verbose=False,
    v_optimal=grids.V_OPTIMAL,
    random_move_choices=grids.RANDOM_MOVE_CHOICES
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
