from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp.common import enums


@dataclasses.dataclass
class GridViewParameters:
    grid_view_type: Optional[enums.GridViewType] = None
    show_demo: Optional[bool] = None
    window_title: Optional[str] = None
    show_trail: Optional[bool] = None
    show_values: Optional[bool] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None

    def apply_default_to_nones(self, default_: GridViewParameters):
        attribute_names: list[str] = [
            'show_demo',
            'window_title',
            'show_trail',
            'show_values',
            'screen_width',
            'screen_height',
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: GridViewParameters = GridViewParameters(
    grid_view_type=enums.GridViewType.POSITION_MOVE,
    show_demo=True,
    window_title="Grid World",
    show_trail=False,
    show_values=True,
    screen_width=1500,
    screen_height=1000,
)


def default_factory() -> GridViewParameters:
    return copy.deepcopy(default)
