from __future__ import annotations
from dataclasses import dataclass

from mdp.common.enums import PolicyType


@dataclass
class PolicyParameters:
    policy_type: PolicyType = PolicyType.TABULAR_E_GREEDY
    epsilon: float = 0.1
    # tabular parameters
    store_matrix: bool = True
    # non-tabular parameters
    initial_theta: float = 0.0      # parameterised policies
    tau: float = 1.0                # softmax temperature parameter
