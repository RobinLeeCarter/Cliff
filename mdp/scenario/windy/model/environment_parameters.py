from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from mdp import common
from mdp.scenario.windy.model import grids
from mdp.scenario.position_move.model.environment_parameters import PositionMoveEnvironmentParameters


@dataclass
class EnvironmentParameters(PositionMoveEnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.WINDY
    actions_list: common.ActionsList = common.ActionsList.FOUR_MOVES
    grid: grids = grids.GRID_1
    upward_wind: np.ndarray = grids.UPWARD_WIND_1
    random_wind: bool = False
    random_wind_choices: np.ndarray = grids.RANDOM_WIND_CHOICES_1
