from __future__ import annotations
from typing import Optional

from mdp import common
from mdp.task.random_walk.model.model import Model
from mdp.task.random_walk.view.view import View

from mdp.task.position_move import controller


class Controller(controller.Controller[Model, View],
                 environment_type=common.EnvironmentType.RANDOM_WALK):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def output(self):
        self._breakdown_2dgraph()

        self._model.update_grid_value_functions()
        self._view.grid_view.display_latest_step()
