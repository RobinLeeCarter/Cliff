from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp.common import enums


@dataclasses.dataclass
class PolicyParameters:
    policy_type: Optional[enums.PolicyType] = None
    epsilon: Optional[float] = None

    def apply_default_to_nones(self, default_: PolicyParameters):
        attribute_names: list[str] = [
            'policy_type',
            'epsilon'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: PolicyParameters = PolicyParameters(
    policy_type=enums.PolicyType.E_GREEDY,
    epsilon=0.1,
)


def default_factory() -> PolicyParameters:
    return copy.deepcopy(default)
