from __future__ import annotations
from dataclasses import dataclass
from mdp import common


@dataclass
class Settings(common.Settings):
    gamma: float = 0.9
    policy_parameters: common.PolicyParameters = common.PolicyParameters(
        policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
    )
    display_every_step: bool = False

