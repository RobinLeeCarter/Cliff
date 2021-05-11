from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.agent.episode import Episode
    from mdp.scenarios.position_move.model.action import Action
    from mdp.scenarios.position_move.model.state import State
    from mdp.scenarios.position_move.model.grid_world import GridWorld

from mdp import common
from mdp.view import grid_view


class GridView(grid_view.GridView):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        super().__init__(grid_view_parameters)
        self._grid_world: Optional[GridWorld] = self._grid_world

    def _frame_on_background_latest(self, episode_: Episode):
        last_state: Optional[State] = episode_.last_s
        agent_position: common.XY = last_state.position
        agent_move: Optional[common.XY] = None
        prev_position: Optional[common.XY] = None
        prev_move: Optional[common.XY] = None

        last_action: Optional[Action] = episode_.last_a
        if last_action:
            agent_move = last_action.move

        prev_state: Optional[State] = episode_.prev_s
        if prev_state:
            prev_position = prev_state.position

        prev_action: Optional[Action] = episode_.prev_a
        if prev_action:
            prev_move = prev_action.move

        self._draw_frame_on_background(agent_position, agent_move, prev_position, prev_move)

    def _frame_on_background_for_t(self, episode_: Episode, t: int):
        state: Optional[State] = episode_[t].state
        agent_position: common.XY = state.position
        agent_move: Optional[common.XY] = None
        prev_position: Optional[common.XY] = None
        prev_move: Optional[common.XY] = None

        last_action: Optional[Action] = episode_[t].action
        if last_action:
            agent_move = last_action.move

        if t >= 1 and not self.grid_view_parameters.show_trail:
            prev_state: Optional[State] = episode_[t - 1].state
            prev_position = prev_state.position
            prev_action: Optional[Action] = episode_[t - 1].action
            if prev_action:
                prev_move: common.XY = prev_action.move

        if self.grid_view_parameters.show_trail:
            self._draw_agent_on_background(agent_position, agent_move)
        else:
            self._copy_grid_into_background()
            self._draw_frame_on_background(agent_position, agent_move, prev_position, prev_move)
