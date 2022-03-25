from __future__ import annotations
from dataclasses import dataclass, field

from mdp.common import enums


@dataclass
class GridViewParameters:
    grid_view_type: enums.GridViewType = enums.GridViewType.POSITION_MOVE
    show_demo: bool = False
    show_result: bool = False
    window_title: str = "Grid World"
    show_trail: bool = False
    
    show_v: bool = False
    show_q: bool = False
    show_policy: bool = False
    
    screen_width: int = 1500
    screen_height: int = 1000

    multi_parameter: list = field(default_factory=list)
