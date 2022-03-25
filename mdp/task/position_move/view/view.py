from __future__ import annotations
from typing import Optional

from mdp.view.tabular.tabular_view import TabularView
from mdp.task.position_move.view.grid_view import GridView


class View(TabularView):
    def __init__(self):
        super().__init__()
        self.grid_view: Optional[GridView] = None

    def _create_grid_view(self) -> GridView:
        return GridView(self._comparison.grid_view_parameters)
