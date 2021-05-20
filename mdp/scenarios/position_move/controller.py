from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.scenarios.position_move.model.model import Model
    from mdp.scenarios.position_move.view.view import View

from mdp import common
from mdp import controller


class Controller(controller.Controller, ABC):
    def __init__(self):
        super().__init__()
        self._model: Optional[Model] = self._model
        self._view: Optional[View] = self._view

    def build(self, comparison: common.Comparison):
        super().build(comparison)
        self._view.grid_view.set_gridworld(self._model.environment.grid_world)
