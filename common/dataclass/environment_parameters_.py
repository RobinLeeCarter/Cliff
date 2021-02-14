from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class EnvironmentParameters:
    environment_type: Optional[enums.EnvironmentType] = None
    actions_list: enums.ActionsList = enums.ActionsList.FOUR_MOVES
    random_wind: bool = False
    verbose: bool = False


def default_factory() -> EnvironmentParameters:
    return EnvironmentParameters()
