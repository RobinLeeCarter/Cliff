from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.general_controller import GeneralController
from mdp.view.general.general_view import GeneralView


class NonTabularView(GeneralView, ABC):
    def set_controller(self, controller: GeneralController):
        self._controller: GeneralController = controller

    def _build(self):
        pass
