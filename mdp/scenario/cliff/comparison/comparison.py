from __future__ import annotations
from dataclasses import dataclass, field

from mdp import common

from mdp.scenario.cliff.comparison.environment_parameters import EnvironmentParameters


@dataclass
class Comparison(common.Comparison):
    # just what is different
    environment_parameters: EnvironmentParameters = field(default_factory=EnvironmentParameters)
