from __future__ import annotations
from dataclasses import dataclass, field

from mdp.common.dataclass import comparison

from mdp.scenario.jacks.model.environment_parameters import EnvironmentParameters


@dataclass
class Comparison(comparison.Comparison):
    environment_parameters: EnvironmentParameters = field(default_factory=EnvironmentParameters)
