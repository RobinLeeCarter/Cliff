from __future__ import annotations
from dataclasses import dataclass
from abc import ABC

from mdp.common.enums import EnvironmentType


@dataclass
class EnvironmentParameters(ABC):
    environment_type: EnvironmentType
    verbose: bool = False
    actions_always_compatible: bool = False
