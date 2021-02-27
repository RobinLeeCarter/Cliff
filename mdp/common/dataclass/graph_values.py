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

    def apply_default_to_nones(self, default_: GraphValues):
        attribute_names: list[str] = [
            'show_graph',
            'title',
            'x_label',
            'y_label',
            'x_min',
            'x_max',
            'y_min',
            'y_max',
            'has_grid',
            'has_legend',
            'moving_average_window_size',
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: GraphValues = GraphValues(
    show_graph=True,
    has_grid=True,
    has_legend=True
)


def default_factory() -> GraphValues:
    return copy.deepcopy(default)
