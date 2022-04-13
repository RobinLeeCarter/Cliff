from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardFeatureVector:
    r: float
    feature_vector: np.ndarray

    @property
    def tuple(self) -> tuple[float, np.ndarray]:
        return self.r, self.feature_vector


FeatureTrajectory = list[RewardFeatureVector]
