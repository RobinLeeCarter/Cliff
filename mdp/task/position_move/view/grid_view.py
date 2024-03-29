from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
    from mdp.task.position_move.model.action import Action
    from mdp.task.position_move.model.grid_world import GridWorld

from mdp.task.position_move.model.state import State
from mdp import common
from mdp.view.tabular.tabular_grid_view import TabularGridView


class GridView(TabularGridView):
    def __init__(self, grid_view_parameters: common.GridViewParameters):
        super().__init__(grid_view_parameters)
        self._grid_world: Optional[GridWorld] = None

    def _frame_on_background_latest(self, episode: TabularEpisode):
        last_state: Optional[State] = episode.last_state
        agent_position: common.XY = last_state.position
        agent_move: Optional[common.XY] = None
        prev_position: Optional[common.XY] = None
        prev_move: Optional[common.XY] = None

        last_action: Optional[Action] = episode.last_action
        if last_action:
            agent_move = last_action.move

        prev_state: Optional[State] = episode.prev_state
        if prev_state:
            prev_position = prev_state.position

        prev_action: Optional[Action] = episode.prev_action
        if prev_action:
            prev_move = prev_action.move

        self._draw_frame_on_background(agent_position, agent_move, prev_position, prev_move)

    def _frame_on_background_for_t(self, episode: TabularEpisode, t: int):
        state: State = episode.get_state(t)  # type: ignore    # Optional[State] before?
        agent_position: common.XY = state.position
        agent_move: Optional[common.XY] = None
        prev_position: Optional[common.XY] = None
        prev_move: Optional[common.XY] = None

        last_action: Optional[Action] = episode.get_action(t)
        if last_action:
            agent_move = last_action.move

        if t >= 1 and not self.grid_view_parameters.show_trail:
            prev_state: State = episode.get_state(t - 1)     # type: ignore
            prev_position = prev_state.position
            prev_action: Optional[Action] = episode.get_action(t - 1)
            if prev_action:
                prev_move: common.XY = prev_action.move

        if self.grid_view_parameters.show_trail:
            self._draw_agent_on_background(agent_position, agent_move)
        else:
            self._copy_grid_into_background()
            self._draw_frame_on_background(agent_position, agent_move, prev_position, prev_move)
