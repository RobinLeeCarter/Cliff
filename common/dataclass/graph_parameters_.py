from __future__ import annotations
import dataclasses
from typing import Optional


@dataclasses.dataclass
class GraphParameters:
    moving_average_window_size: Optional[int] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


def default_factory() -> GraphParameters:
    return GraphParameters()
