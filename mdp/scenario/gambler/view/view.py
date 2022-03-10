from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenario.gambler.controller import Controller

from mdp.view.general_view import GeneralView


class View(GeneralView):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = self._controller
