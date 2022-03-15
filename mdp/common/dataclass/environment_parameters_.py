from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from abc import ABC

import numpy as np

from mdp.common import enums


@dataclass
class EnvironmentParameters(ABC):
    environment_type: Optional[enums.EnvironmentType] = None
    grid: Optional[np.ndarray] = None
    actions_list: Optional[enums.ActionsList] = None
    actions_always_compatible: Optional[bool] = None
    verbose: Optional[bool] = None


# def none_factory() -> EnvironmentParameters:
#     return EnvironmentParameters()

