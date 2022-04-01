from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.task.gambler.controller import Controller
from mdp import common
from mdp.view.tabular.tabular_view import TabularView


class View(TabularView,
           environment_type=common.EnvironmentType.GAMBLER):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = None
