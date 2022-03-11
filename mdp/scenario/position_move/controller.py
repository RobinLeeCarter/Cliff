from __future__ import annotations
from typing import Optional, TypeVar
from abc import ABC

from mdp.scenario.position_move.model import model
from mdp.scenario.position_move.view import view

from mdp import common
from mdp.controller.tabular_controller import TabularController


Model = TypeVar("Model", bound=model.Model)
View = TypeVar("View", bound=view.View)


class Controller(TabularController[Model, View], ABC):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self._view.grid_view.set_gridworld(self._model.environment.grid_world)
