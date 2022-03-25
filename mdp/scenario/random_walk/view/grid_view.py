from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenario.random_walk.model.grid_world import GridWorld


from mdp import common
from mdp.view.tabular.tabular_grid_view import TabularGridView


class GridView(TabularGridView):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        super().__init__(grid_view_parameters)
        self._grid_world: Optional[GridWorld] = None
