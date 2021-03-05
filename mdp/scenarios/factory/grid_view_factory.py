from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.view import grid_view as base_grid_view
from mdp import common
from mdp.scenarios.position_move import grid_view as pm


def grid_view_factory(grid_view_parameters: common.GridViewParameters) -> base_grid_view.GridView:
    grid_view_type = grid_view_parameters.grid_view_type
    gvt = common.GridViewType

    if grid_view_type == gvt.POSITION_MOVE:
        grid_view_ = pm.GridView(grid_view_parameters)
    else:
        raise NotImplementedError
    return grid_view_