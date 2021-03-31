from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp import common

from mdp.view import view
from mdp.scenarios.blackjack.view.grid_view import GridView


class View(view.View):
    def __init__(self):
        super().__init__()
        self.grid_view: Optional[GridView] = self.grid_view

    def _create_grid_view(self, grid_view_parameters: common.GridViewParameters) -> GridView:
        return GridView(grid_view_parameters)
