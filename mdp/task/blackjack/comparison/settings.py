from __future__ import annotations
from dataclasses import dataclass
from mdp import common


@dataclass
class Settings(common.Settings):
    gamma: float = 1.0
    runs: int = 1
    training_episodes: int = 500_000
    episode_print_frequency: int = 10_000
    policy_parameters: common.PolicyParameters = common.PolicyParameters(
        policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
    )
