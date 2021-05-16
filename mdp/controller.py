from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model.agent.episode import Episode
    from mdp.model.model import Model
    from mdp.view.view import View
    from mdp import common


class Controller:
    def __init__(self):
        self._model: Optional[Model] = None
        self._view: Optional[View] = None
        self._comparison: Optional[common.Comparison] = None

    def link_mvc(self, model: Model, view: View):
        self._model: Model = model
        self._view: View = view
        self._model.set_controller(self)
        self._view.set_controller(self)

    def build(self, comparison: common.Comparison):
        # self._view.open() to determine user environment only
        self._comparison = comparison
        self._model.build(self._comparison)
        # TODO: don't pass grid_world here - too specific
        self._view.build(grid_world_=self._model.environment.grid_world, comparison=self._comparison)

    def run(self):
        import cProfile
        cProfile.runctx('self._model.run()', globals(), locals(), 'model_run.prof')
        # self._model.run()

    def output(self):
        pass

    def _breakdown_graph(self):
        graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
        self._view.graph.make_plot(graph_values)

    # region Model requests
    # def display_graph_2d(self, graph_values: common.GraphValues):
    #     self._view.graph.make_plot(graph_values)

    def display_step(self, episode_: Optional[Episode]):
        # if self._comparison.grid_view_parameters.show_step:
        self._view.grid_view.display_latest_step(episode_)
    # endregion

    # region View requests
    def new_episode_request(self) -> Episode:
        return self._model.agent.generate_episode()
    # endregion
