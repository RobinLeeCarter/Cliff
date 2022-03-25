from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from abc import ABC

import numpy as np

from mdp import common


@dataclass
class TabularEnvironmentParameters(common.EnvironmentParameters, ABC):
    grid: Optional[np.ndarray] = None
