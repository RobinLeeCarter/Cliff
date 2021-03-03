from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import agent
    from mdp.model.scenarios.common import state_position, action_move

from mdp import common
from mdp.view import grid_view


class GridView(grid_view.GridView):
    def _frame_on_background_latest(self, episode_: agent.Episode):
        last_state: state_position.State = episode_.last_state
        agent_position: common.XY = last_state.position
        agent_move: Optional[common.XY] = None
        prev_position: Optional[common.XY] = None
        prev_move: Optional[common.XY] = None

        last_action: Optional[action_move.Action] = episode_.last_action
        if last_action:
            agent_move = last_action.move

        prev_state: Optional[state_position.State] = episode_.prev_state
        if prev_state:
            prev_position = prev_state.position

        prev_action: Optional[action_move.Action] = episode_.prev_action
        if prev_action:
            prev_move = prev_action.move

        self._draw_frame_on_background(agent_position, agent_move, prev_position, prev_move)

    def _frame_on_background_for_t(self, episode_: agent.Episode, t: int):
        state: state_position.State = episode_[t].state
        agent_position: common.XY = state.position
        agent_move: Optional[common.XY] = None
        prev_position: Optional[common.XY] = None
        prev_move: Optional[common.XY] = None

        last_action: Optional[action_move.Action] = episode_[t].action
        if last_action:
            agent_move = last_action.move

        if t >= 1 and not self.grid_view_parameters.show_trail:
            prev_state: state_position.State = episode_[t-1].state
            prev_position = prev_state.position
            prev_action: Optional[action_move.Action] = episode_[t-1].action
            if prev_action:
                prev_move: common.XY = prev_action.move

        if self.grid_view_parameters.show_trail:
            self._draw_agent_on_background(agent_position, agent_move)
        else:
            self._copy_grid_into_background()
            self._draw_frame_on_background(agent_position, agent_move, prev_position, prev_move)
