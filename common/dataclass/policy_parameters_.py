from __future__ import annotations
import dataclasses
from typing import Optional


@dataclasses.dataclass
class PolicyParameters:
    epsilon: Optional[float] = None


precedence_attribute_names: list[str] = [
    'epsilon'
]


def default_factory() -> PolicyParameters:
    return PolicyParameters()
