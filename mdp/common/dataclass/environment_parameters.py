from __future__ import annotations
from typing import Optional
import dataclasses
import copy


from mdp.common import enums


@dataclasses.dataclass
class EnvironmentParameters:
    environment_type: Optional[enums.EnvironmentType] = None
    actions_list: Optional[enums.ActionsList] = None
    random_wind: Optional[bool] = None
    verbose: Optional[bool] = None


default: EnvironmentParameters = EnvironmentParameters(
    actions_list=enums.ActionsList.FOUR_MOVES,
    random_wind=False,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
