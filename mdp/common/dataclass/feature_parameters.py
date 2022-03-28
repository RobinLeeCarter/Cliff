from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from abc import ABC

from mdp.common.enums import FeatureType


@dataclass
class FeatureParameters(ABC):
    feature_type: FeatureType
    max_size: Optional[int] = None
