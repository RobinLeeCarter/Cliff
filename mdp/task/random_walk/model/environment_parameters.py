from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.task.random_walk.model import grids
from mdp.task._position_move.model.environment_parameters import PositionMoveEnvironmentParameters


@dataclass
class EnvironmentParameters(PositionMoveEnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.RANDOM_WALK
    actions_list: common.ActionsList = common.ActionsList.NO_ACTIONS
    grid: grids = grids.GRID
    v_optimal: grids = grids.V_OPTIMAL
    random_move_choices: grids = grids.RANDOM_MOVE_CHOICES
