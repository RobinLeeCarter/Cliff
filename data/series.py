import dataclasses

import numpy as np


@dataclasses.dataclass
class Series:
    title: str
    identifiers: dict = dataclasses.field(default_factory=dict)
    values: np.ndarray = np.array([], dtype=float)
