from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.view.grid_view import GridView as BaseGridView
from mdp import common
from mdp.scenarios.position.grid_view import GridView as PositionGridView
from mdp.scenarios.position_move.grid_view import GridView as PositionMoveGridView
from mdp.scenarios.jacks.view.grid_view import GridView as JacksGridView
from mdp.scenarios.blackjack.view.grid_view import GridView as BlackjackGridView


def grid_view_factory(grid_view_parameters: common.GridViewParameters) -> BaseGridView:
    grid_view_type = grid_view_parameters.grid_view_type
    gvt = common.GridViewType

    if grid_view_type == gvt.POSITION:
        grid_view_ = PositionGridView(grid_view_parameters)
    elif grid_view_type == gvt.POSITION_MOVE:
        grid_view_ = PositionMoveGridView(grid_view_parameters)
    elif grid_view_type == gvt.JACKS:
        grid_view_ = JacksGridView(grid_view_parameters)
    elif grid_view_type == gvt.BLACKJACK:
        grid_view_ = BlackjackGridView(grid_view_parameters)
    else:
        raise NotImplementedError
    return grid_view_
