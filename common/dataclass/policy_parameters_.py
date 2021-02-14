from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class PolicyParameters:
    policy_type: Optional[enums.PolicyType] = None
    epsilon: Optional[float] = None


precedence_attribute_names: list[str] = [
    'policy_type',
    'epsilon'
]


def default_factory() -> PolicyParameters:
    return PolicyParameters()
