from __future__ import annotations
import dataclasses
import copy

from mdp import common

from mdp.scenarios.cliff import grids


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    pass


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.CLIFF,
    actions_list=common.ActionsList.FOUR_MOVES,
    grid=grids.CLIFF_1,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
