from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from abc import ABC

import numpy as np

from mdp.common import enums


@dataclass
class EnvironmentParameters(ABC):
    environment_type: enums.EnvironmentType
    verbose: bool = False
    actions_always_compatible: bool = False

    # TODO: Make only for some scenarios
    grid: Optional[np.ndarray] = None
    actions_list: Optional[enums.ActionsList] = None
