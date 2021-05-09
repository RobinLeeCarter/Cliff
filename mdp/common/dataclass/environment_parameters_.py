from __future__ import annotations
from typing import Optional
import dataclasses
import abc

import numpy as np

from mdp.common import enums


@dataclasses.dataclass
class EnvironmentParameters(abc.ABC):
    environment_type: Optional[enums.ScenarioType] = None
    grid: Optional[np.ndarray] = None
    actions_list: Optional[enums.ActionsList] = None
    verbose: Optional[bool] = None


# def none_factory() -> EnvironmentParameters:
#     return EnvironmentParameters()

