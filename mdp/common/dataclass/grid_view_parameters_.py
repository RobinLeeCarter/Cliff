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
    
    show_v: Optional[bool] = None
    show_q: Optional[bool] = None
    show_policy: Optional[bool] = None
    
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None

    multi_parameter: list = dataclasses.field(default_factory=list)


default: GridViewParameters = GridViewParameters(
    grid_view_type=enums.GridViewType.POSITION_MOVE,
    show_demo=False,
    show_result=False,
    window_title="Grid World",
    show_trail=False,
    show_v=False,
    show_q=False,
    show_policy=False,
    screen_width=1500,
    screen_height=1000,
)


def default_factory() -> GridViewParameters:
    return copy.deepcopy(default)
