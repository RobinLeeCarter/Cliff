from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import copy

from mdp.common import enums


@dataclass
class BreakdownParameters:
    breakdown_type: Optional[enums.BreakdownType] = None
    verbose: Optional[bool] = None


default: BreakdownParameters = BreakdownParameters(
    verbose=False
)


def default_factory() -> BreakdownParameters:
    return copy.deepcopy(default)
