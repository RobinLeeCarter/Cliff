from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp import common


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    max_cars: Optional[int] = None
    max_transfers: Optional[int] = None


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.JACKS,
    max_cars=20,
    max_transfers=5,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
