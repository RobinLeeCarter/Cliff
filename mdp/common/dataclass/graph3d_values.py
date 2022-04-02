from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from mdp.common.dataclass.series import Series


@dataclass
class Graph3DValues:
    x_series: Optional[Series] = None    # 1d
    y_series: Optional[Series] = None    # 1d
    z_series: Optional[Series] = None    # 2d z=f(x,y)

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

    has_grid: bool = False
    has_legend: bool = False
    multi_parameter: list = field(default_factory=list)
    steps: Optional[int] = None
