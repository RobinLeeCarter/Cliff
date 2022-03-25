from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar, Generic

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.base.agent.base_episode import BaseEpisode

from mdp.model.base.base_model import BaseModel
from mdp.view.base.base_view import BaseView

Model = TypeVar("Model", bound=BaseModel)
View = TypeVar("View", bound=BaseView)


class BaseController(Generic[Model, View]):
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
        self._view.build(self._comparison)

    def run(self):
        # import cProfile
        # cProfile.runctx('self._model.run()', globals(), locals(), 'model_run.prof')
        self._model.run()

    def output(self):
        pass

    def _breakdown_2dgraph(self):
        graph_values: common.Graph2DValues = self._model.breakdown.get_graph2d_values()
        self._view.graph2d.make_plot(graph_values)

    # region Model requests
    # def display_graph_2d(self, graph_values: common.GraphValues):
    #     self._view.graph.make_plot(graph_values)

    def display_step(self, episode: Optional[BaseEpisode]):
        raise Exception("display_step() Not Implemented")
    # endregion

    # region View requests
    def new_episode_request(self) -> BaseEpisode:
        raise Exception("new_episode_request() Not Implemented")
    # endregion
