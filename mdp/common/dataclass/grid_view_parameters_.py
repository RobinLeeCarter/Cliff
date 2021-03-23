from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from mdp.common import enums


@dataclasses.dataclass
class GridViewParameters:
    grid_view_type: Optional[enums.GridViewType] = None
    show_demo: Optional[bool] = None
    show_result: Optional[bool] = None
    window_title: Optional[str] = None
    show_trail: Optional[bool] = None
    show_values: Optional[bool] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None


default: GridViewParameters = GridViewParameters(
    grid_view_type=enums.GridViewType.POSITION_MOVE,
    show_demo=True,
    show_result=False,
    window_title="Grid World",
    show_trail=False,
    show_values=True,
    screen_width=1500,
    screen_height=1000,
)


def default_factory() -> GridViewParameters:
    return copy.deepcopy(default)
