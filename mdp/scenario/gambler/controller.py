from __future__ import annotations
from typing import Optional

from mdp.scenario.gambler.model.model import Model
from mdp.scenario.gambler.view.view import View
from mdp.controller.tabular_controller import TabularController


class Controller(TabularController[Model, View]):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model

    def output(self):
        self._model.environment.insert_state_function_into_graph2d(
            self._comparison,
            self._model.algorithm.V
        )
        self._view.graph2d.make_plot(self._comparison.graph2d_values)

        self._model.environment.insert_policy_into_graph2d(
            self._comparison,
            self._model.algorithm.target_policy
        )
        self._view.graph2d.make_plot(self._comparison.graph2d_values)
