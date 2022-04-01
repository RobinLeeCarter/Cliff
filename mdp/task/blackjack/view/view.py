from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from mdp import common

if TYPE_CHECKING:
    from mdp.task.jacks.controller import Controller

from mdp.view.tabular.tabular_view import TabularView
from mdp.task.blackjack.view.grid_view import GridView


class View(TabularView,
           environment_type=common.EnvironmentType.BLACKJACK):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = None
        self.grid_view: Optional[GridView] = None

    def _create_grid_view(self) -> GridView:
        return GridView(self._comparison.grid_view_parameters)
