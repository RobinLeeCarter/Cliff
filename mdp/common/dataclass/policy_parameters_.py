from __future__ import annotations
from typing import Optional
from dataclasses import dataclass


from mdp.common import enums


@dataclass
class PolicyParameters:
    policy_type: Optional[enums.PolicyType] = None
    epsilon: Optional[float] = None
    store_matrix: Optional[bool] = None


default: PolicyParameters = PolicyParameters(
    policy_type=enums.PolicyType.TABULAR_E_GREEDY,
    epsilon=0.1,
    store_matrix=True,
)


def none_factory() -> PolicyParameters:
    return PolicyParameters()
