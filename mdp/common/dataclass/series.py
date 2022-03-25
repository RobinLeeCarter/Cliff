from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Series:
    title: str
    identifiers: dict = field(default_factory=dict)
    values: np.ndarray = np.array([], dtype=float)
