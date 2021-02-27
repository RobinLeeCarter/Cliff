from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class GridViewParameters:
    window_title: str = "Grid World"
    show_trail: bool = False
    show_values: bool = True
    screen_width: int = 1500
    screen_height: int = 1000
    cell_pixels: int = 10


def default_factory() -> GridViewParameters:
    return GridViewParameters()
