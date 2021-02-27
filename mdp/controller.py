from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import agent

from mdp import common, model, view


class Controller:
    def __init__(self, model_: model.Model, view_: view.View):
        self._model: model.Model = model_
        self._view: view.View = view_

    def build(self, comparison: common.Comparison):
        # self._view.open() to determine user environment only
        self._model.build(comparison)
        self._view.build(grid_world_=self._model.environment.grid_world, comparison=comparison)

    def run(self):
        self._model.run()
        self._model.update_grid_value_functions()
        graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
        self._view.graph.make_plot(graph_values)
        self._view.grid_view.demonstrate(self.new_episode_request)

    # region Model requests
    def display_step(self, episode_: agent.Episode):
        self._view.grid_view.display_step(episode_)
    # endregion

    # region View requests
    def new_episode_request(self) -> agent.Episode:
        return self._model.agent.generate_episode()
    # endregion
