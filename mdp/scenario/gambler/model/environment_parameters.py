from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.model.tabular.environment.tabular_environment_parameters import TabularEnvironmentParameters


@dataclass
class EnvironmentParameters(TabularEnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.GAMBLER
    probability_heads: float = 0.4
    max_capital: int = 100
