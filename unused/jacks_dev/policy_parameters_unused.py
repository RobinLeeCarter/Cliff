from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import copy

from mdp import common


@dataclass
class PolicyParameters(common.PolicyParameters):
    policy_type: Optional[common.PolicyType] = None
    epsilon: Optional[float] = None


default: PolicyParameters = PolicyParameters(
    policy_type=common.PolicyType.TABULAR_DETERMINISTIC,
)


def default_factory() -> PolicyParameters:
    return copy.deepcopy(default)
