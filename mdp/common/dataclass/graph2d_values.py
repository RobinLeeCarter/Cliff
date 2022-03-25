from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from mdp.common.dataclass.series import Series


@dataclass
class Graph2DValues:
    x_series: Optional[Series] = None
    graph_series: list[Series] = field(default_factory=list)    # 2d Yi=Fi(x)

    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    has_grid: bool = False
    has_legend: bool = False

    moving_average_window_size: Optional[int] = None
