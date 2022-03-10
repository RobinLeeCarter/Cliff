from __future__ import annotations
from typing import Optional

from mdp.view import view
from mdp.scenario.position_move.view.grid_view import GridView


class View(view.View):
    def __init__(self):
        super().__init__()
        self.grid_view: Optional[GridView] = self.grid_view

    def _create_grid_view(self) -> GridView:
        return GridView(self._comparison.grid_view_parameters)
