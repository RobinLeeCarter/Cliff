from __future__ import annotations
from dataclasses import dataclass

from mdp.common.enums import PolicyType


@dataclass
class PolicyParameters:
    policy_type: PolicyType = PolicyType.TABULAR_E_GREEDY
    epsilon: float = 0.1
    store_matrix: bool = True
