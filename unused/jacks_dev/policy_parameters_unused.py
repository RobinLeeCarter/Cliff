from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp import common


@dataclasses.dataclass
class PolicyParameters(common.PolicyParameters):
    policy_type: Optional[common.PolicyType] = None
    epsilon: Optional[float] = None


default: PolicyParameters = PolicyParameters(
    policy_type=common.PolicyType.DETERMINISTIC,
)


def default_factory() -> PolicyParameters:
    return copy.deepcopy(default)
