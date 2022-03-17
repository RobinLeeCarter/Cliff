from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from mdp.common.enums import BreakdownType


@dataclass
class BreakdownParameters:
    breakdown_type: Optional[BreakdownType] = None
    verbose: bool = False
