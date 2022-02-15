from __future__ import annotations
# from typing import Optional
import dataclasses
import copy

from mdp import common


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    pass


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.ScenarioType.MOUNTAIN_CAR,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
