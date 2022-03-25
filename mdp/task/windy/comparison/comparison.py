from __future__ import annotations
from dataclasses import dataclass

from mdp.common.dataclass import comparison


@dataclass
class Comparison(comparison.Comparison):
    environment_parameters: EnvironmentParametersmax_cars
