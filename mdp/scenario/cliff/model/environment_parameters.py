from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.scenario.cliff.model import grids
from mdp.scenario.position_move.model.environment_parameters import PositionMoveEnvironmentParameters


@dataclass
class EnvironmentParameters(PositionMoveEnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.CLIFF
    actions_list: common.ActionsList = common.ActionsList.FOUR_MOVES
    grid: grids = grids.CLIFF_1
