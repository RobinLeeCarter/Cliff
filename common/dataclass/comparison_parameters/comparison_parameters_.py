from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class ComparisonParameters:
    comparison_type: Optional[enums.ComparisonType] = None
    verbose: bool = False


def default_factory() -> ComparisonParameters:
    return ComparisonParameters()
