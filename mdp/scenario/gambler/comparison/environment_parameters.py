from __future__ import annotations
from dataclasses import dataclass

from mdp import common


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.GAMBLER
    probability_heads: float = 0.4
    max_capital: int = 100
