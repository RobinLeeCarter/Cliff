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
        # print(self._model.agent.algorithm.V)

        if self._comparison.graph_values.show_graph:
            graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
            self._view.graph.make_plot(graph_values)

        if self._comparison.graph3d_values.show_graph:
            self._model.environment.insert_state_function_into_graph3d(
                self._comparison,
                self._model.agent.algorithm.V
            )
            self._view.graph3d.make_plot(self._comparison.graph3d_values)

        gvp = self._comparison.grid_view_parameters
        if gvp.show_result or gvp.show_demo:
            self._model.prep_for_output()
            if gvp.show_result:
                self._view.grid_view.display_latest_step()
            if gvp.show_demo:
                self._view.grid_view.demonstrate(self.new_episode_request)

    # region Model requests
    def display_step(self, episode_: Optional[agent.Episode]):
        # if self._comparison.grid_view_parameters.show_step:
        self._view.grid_view.display_latest_step(episode_)
    # endregion

    # region View requests
    def new_episode_request(self) -> agent.Episode:
        return self._model.agent.generate_episode()
    # endregion
