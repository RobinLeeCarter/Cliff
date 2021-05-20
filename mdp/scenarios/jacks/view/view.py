from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenarios.jacks.controller import Controller

from mdp.view import view
from mdp.scenarios.jacks.view.grid_view import GridView


class View(view.View):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = self._controller
        self.grid_view: Optional[GridView] = self.grid_view

    def _create_grid_view(self) -> GridView:
        return GridView(self._comparison.grid_view_parameters)
