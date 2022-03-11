from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model.general.agent.general_episode import GeneralEpisode
    from mdp.model.general.general_model import GeneralModel
    from mdp.view.tabular.tabular_view import TabularView
    from mdp import common


class GeneralController2:
    def __init__(self):
        self._model: Optional[GeneralModel] = None
        self._view: Optional[TabularView] = None
        self._comparison: Optional[common.Comparison] = None

    def link_mvc(self, model: GeneralModel, view: TabularView):
        self._model: GeneralModel = model
        self._view: TabularView = view
        self._model.set_controller(self)
        self._view.set_controller(self)

    def build(self, comparison: common.Comparison):
        # self._view.open() to determine user environment only
        self._comparison = comparison
        self._model.build(self._comparison)
        self._view.build(self._comparison)

    def run(self):
        # import cProfile
        # cProfile.runctx('self._model.run()', globals(), locals(), 'model_run.prof')
        self._model.run()

    def output(self):
        pass

    def _breakdown_graph(self):
        graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
        self._view.graph2d.make_plot(graph_values)

    # region Model requests
    # def display_graph_2d(self, graph_values: common.GraphValues):
    #     self._view.graph.make_plot(graph_values)

    def display_step(self, episode_: Optional[GeneralEpisode]):
        # if self._comparison.grid_view_parameters.show_step:
        self._view.grid_view.display_latest_step(episode_)
    # endregion

    # region View requests
    def new_episode_request(self) -> GeneralEpisode:
        return self._model.agent.generate_episode()
    # endregion
