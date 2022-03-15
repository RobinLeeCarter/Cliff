from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.scenario.cliff.model import grids


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.CLIFF
    actions_list: common.ActionsList = common.ActionsList.FOUR_MOVES
    grid: grids = grids.CLIFF_1
