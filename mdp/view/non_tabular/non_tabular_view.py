from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.controller.tabular_controller import TabularController
from mdp.view.general.general_view import GeneralView


class NonTabularView(GeneralView, ABC):
    def set_controller(self, controller: TabularController):
        self._controller: TabularController = controller
