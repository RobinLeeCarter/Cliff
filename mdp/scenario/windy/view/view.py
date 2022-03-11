from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.scenario.windy.controller import Controller

from mdp.scenario.position_move.view import view


class View(view.View):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = None
