from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model import agent

from mdp import common
from mdp.model import model_
from mdp.view import view_


class Controller:
    def __init__(self, model: model_.Model, view: view_.View):
        self._model: model_.Model = model
        self._view: view_.View = view
        self._comparison: Optional[common.Comparison] = None

    def build(self, comparison: common.Comparison):
        # self._view.open() to determine user environment only
        self._comparison = comparison
        self._model.build(self._comparison)
        self._view.build(grid_world_=self._model.environment.grid_world, comparison=self._comparison)

    def run(self):
        self._model.run()

        if self._comparison.graph_values.show_graph:
            graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
            self._view.graph.make_plot(graph_values)

        if self._comparison.grid_view_parameters.show_demo:
            self._model.update_grid_value_functions()
            self._view.grid_view.demonstrate(self.new_episode_request)

    # region Model requests
    def display_step(self, episode_: agent.Episode):
        self._view.grid_view.display_step(episode_)
    # endregion

    # region View requests
    def new_episode_request(self) -> agent.Episode:
        return self._model.agent.generate_episode()
    # endregion
