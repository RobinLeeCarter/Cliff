from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenarios.windy.controller import Controller

from mdp.scenarios.position_move.view import view


class View(view.View):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = self._controller
