from __future__ import annotations
from dataclasses import dataclass

from mdp import common


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    environment_type: common.EnvironmentType = common.EnvironmentType.BLACKJACK
