from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenario.random_walk.model.grid_world import GridWorld


from mdp import common
from mdp.view.tabular import grid_view


class GridView(grid_view.GridView):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        super().__init__(grid_view_parameters)
        self._grid_world: Optional[GridWorld] = self._grid_world
