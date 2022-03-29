from __future__ import annotations
from typing import Optional

from mdp.task.windy.model.model import Model
from mdp.task.windy.view.view import View

from mdp.task._position_move import controller


class Controller(controller.Controller[Model, View]):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def output(self):
        self._breakdown_2dgraph()

        grid_view_parameters = self._comparison.grid_view_parameters
        if grid_view_parameters and grid_view_parameters.show_demo:
            self._model.update_grid_value_functions()
            self._view.grid_view.demonstrate(self.new_episode_request)
