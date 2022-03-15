from __future__ import annotations
# from typing import Optional
from dataclasses import dataclass
import copy

from mdp import common


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    pass


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.MOUNTAIN_CAR,
    actions_always_compatible=True,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
