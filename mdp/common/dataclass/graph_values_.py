from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp.common.dataclass import series


@dataclasses.dataclass
class GraphValues:
    x_series: Optional[series.Series] = None
    graph_series: list[series.Series] = dataclasses.field(default_factory=list)

    show_graph: Optional[bool] = None
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    has_grid: Optional[bool] = None
    has_legend: Optional[bool] = None

    moving_average_window_size: Optional[int] = None


default: GraphValues = GraphValues(
    show_graph=False,
    has_grid=False,
    has_legend=False
)


def default_factory() -> GraphValues:
    return copy.deepcopy(default)
