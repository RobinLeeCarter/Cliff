from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.agent.episode import Episode
    from mdp.scenarios.position.model.state import State

from mdp import common
from mdp.view import grid_view


class GridView(grid_view.GridView):
    def _frame_on_background_latest(self, episode_: Episode):
        last_state: Optional[State] = episode_.last_state
        agent_position: common.XY = last_state.position
        prev_position: Optional[common.XY] = None

        prev_state: Optional[State] = episode_.prev_state
        if prev_state:
            prev_position = prev_state.position

        self._draw_frame_on_background(agent_position=agent_position, prev_position=prev_position)

    def _frame_on_background_for_t(self, episode: Episode, t: int):
        state: Optional[State] = episode[t].state
        agent_position: common.XY = state.position
        prev_position: Optional[common.XY] = None

        if t >= 1 and not self.grid_view_parameters.show_trail:
            prev_state: Optional[State] = episode[t - 1].state
            prev_position = prev_state.position

        if self._grid_world.is_inside(agent_position):
            if self.grid_view_parameters.show_trail:
                self._draw_agent_on_background(agent_position=agent_position)
            elif self._grid_world.is_inside(prev_position):
                self._copy_grid_into_background()
                self._draw_frame_on_background(agent_position=agent_position, prev_position=prev_position)
