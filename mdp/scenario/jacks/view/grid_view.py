from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenario.jacks.model.grid_world import GridWorld

import pygame
from matplotlib import colors

from mdp import common
from mdp.view.tabular.tabular_grid_view import TabularGridView


class GridView(TabularGridView):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        super().__init__(grid_view_parameters)
        self._grid_world: Optional[GridWorld] = None

    def set_gridworld(self, grid_world: GridWorld):
        super().set_gridworld(grid_world)
        self._policy_color_normaliser = colors.Normalize(
            vmin=self._grid_world.policy_min,
            vmax=self._grid_world.policy_max
        )

    def _draw_policy(self, surface: pygame.Surface, rect: pygame.Rect, output_square: common.OutputSquare):
        policy_value = output_square.policy_value
        if policy_value is not None:
            policy_color: pygame.Color = self._get_policy_value_color(policy_value)
            pygame.draw.rect(surface, policy_color, rect)

            text: str = f"{policy_value:.0f}"
            sub_rect = self._get_sub_rect(rect, move=common.XY(x=0, y=0))
            self._center_text(surface, sub_rect, text)
