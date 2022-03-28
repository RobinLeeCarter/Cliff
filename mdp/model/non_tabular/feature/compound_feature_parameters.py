from __future__ import annotations
from dataclasses import dataclass, field

from mdp import common


@dataclass
class CompoundFeatureParameters(common.FeatureParameters):
    features: list[common.FeatureParameters] = field(default_factory=list)
