from __future__ import annotations
from typing import Optional

from mdp import common
from mdp.controller.tabular_controller import TabularController
from mdp.scenario.blackjack.model.model import Model
from mdp.scenario.blackjack.view.view import View


class Controller(TabularController[Model, View]):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = None
        self._view: Optional[View] = None

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self._view.grid_view.set_gridworld(self._model.environment.grid_world)

    def output(self):
        for usable_ace in [False, True]:
            self._model.environment.update_grid_policy_ace(self._model.algorithm, usable_ace)
            self._view.grid_view.set_title(usable_ace)
            self._view.grid_view.display_latest_step()

            if self._comparison.graph3d_values:
                graph3d_values = self._model.get_state_function_graph(usable_ace)
                self._view.graph3d.make_plot(graph3d_values)
