from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp.common.dataclass import series


@dataclasses.dataclass
class Graph3DValues:
    x_series: Optional[series.Series] = None    # 1d
    y_series: Optional[series.Series] = None    # 1d
    z_series: Optional[series.Series] = None    # 2d z=f(x,y)

    show_graph: Optional[bool] = None
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    z_label: Optional[str] = None

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None

    has_grid: Optional[bool] = None
    has_legend: Optional[bool] = None
    multi_parameter: list = dataclasses.field(default_factory=list)


default: Graph3DValues = Graph3DValues(
    show_graph=False,
    has_grid=False,
    has_legend=False
)


def default_factory() -> Graph3DValues:
    return copy.deepcopy(default)
