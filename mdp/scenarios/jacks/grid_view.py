from __future__ import annotations
from typing import TYPE_CHECKING, Optional  # , Optional

if TYPE_CHECKING:
    from mdp.model import agent, environment
    from mdp.scenarios.jacks.grid_world import GridWorld
    # from mdp.scenarios.position.state import State

import pygame
from matplotlib import colors

from mdp import common
from mdp.view import grid_view


class GridView(grid_view.GridView):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        super().__init__(grid_view_parameters)
        self._grid_world: Optional[GridWorld] = self._grid_world

    def set_gridworld(self, grid_world: environment.GridWorld):
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

    def _frame_on_background_latest(self, episode_: agent.Episode):
        raise NotImplementedError
        # self._draw_frame_on_background()

    def _frame_on_background_for_t(self, episode: agent.Episode, t: int):
        raise NotImplementedError
        # state: Optional[State] = episode[t].state
        # agent_position: common.XY = state.position
        # prev_position: Optional[common.XY] = None
        #
        # if t >= 1 and not self.grid_view_parameters.show_trail:
        #     prev_state: Optional[State] = episode[t - 1].state
        #     prev_position = prev_state.position
        #
        # if self._grid_world.is_inside(agent_position):
        #     if self.grid_view_parameters.show_trail:
        #         self._draw_agent_on_background(agent_position=agent_position)
        #     elif self._grid_world.is_inside(prev_position):
        #         self._copy_grid_into_background()
        #         self._draw_frame_on_background(agent_position=agent_position, prev_position=prev_position)
