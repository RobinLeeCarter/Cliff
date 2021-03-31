from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenarios.racetrack.model.model import Model
    from mdp.scenarios.racetrack.view.view import View

from mdp import common
from mdp import controller


class Controller(controller.Controller):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def output(self):
        graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
        self._view.graph.make_plot(graph_values)

        self._model.prep_for_output()
        self._view.grid_view.demonstrate(self.new_episode_request)
