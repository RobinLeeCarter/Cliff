from __future__ import annotations
from typing import Optional

from mdp.scenario.random_walk.model.model import Model
from mdp.scenario.random_walk.view.view import View

from mdp.scenario.position_move import controller


class Controller(controller.Controller[Model, View]):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def output(self):
        self._breakdown_graph()

        self._model.update_grid_value_functions()
        self._view.grid_view.display_latest_step()
