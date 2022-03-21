from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.controller.tabular_controller import TabularController
from mdp.view.base.base_view import BaseView


class NonTabularView(BaseView, ABC):
    def set_controller(self, controller: TabularController):
        self._controller: TabularController = controller
