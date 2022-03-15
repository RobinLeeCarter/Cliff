from __future__ import annotations
from dataclasses import dataclass, field

from mdp.common.dataclass import comparison

from mdp.scenario.gambler.comparison.environment_parameters import EnvironmentParameters


@dataclass
class Comparison(comparison.Comparison):
    # just what is different
    environment_parameters: EnvironmentParameters = field(default_factory=EnvironmentParameters)
