from __future__ import annotations


from typing import Optional

from mdp.task.gambler.model.model import Model
from mdp.task.gambler.view.view import View
from mdp.controller.tabular_controller import TabularController


class Controller(TabularController[Model, View]):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model

    def output(self):
        if self._comparison.graph2d_values:
            graph2d_values = self._model.get_state_graph_values()
            self._view.graph2d.make_plot(graph2d_values)

            graph2d_values = self._model.get_policy_graph_values()
            self._view.graph2d.make_plot(graph2d_values)
