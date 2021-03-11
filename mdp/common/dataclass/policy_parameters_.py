from __future__ import annotations
from typing import Optional   # , TYPE_CHECKING
import dataclasses

# if TYPE_CHECKING:
#     from mdp.model import policy

from mdp.common import enums


@dataclasses.dataclass
class PolicyParameters:
    policy_type: Optional[enums.PolicyType] = None
    initialize: Optional[bool] = None
    epsilon: Optional[float] = None


default: PolicyParameters = PolicyParameters(
    policy_type=enums.PolicyType.E_GREEDY,
    initialize=False,
    epsilon=0.1,
)


def none_factory() -> PolicyParameters:
    return PolicyParameters()
