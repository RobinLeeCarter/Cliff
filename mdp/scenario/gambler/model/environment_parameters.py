from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import copy

from mdp import common


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    probability_heads: Optional[float] = None
    max_capital: Optional[int] = None


default: EnvironmentParameters = EnvironmentParameters(
    environment_type=common.EnvironmentType.GAMBLER,
    probability_heads=0.4,
    max_capital=100,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
