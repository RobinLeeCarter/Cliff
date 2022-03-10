from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp.scenario.position_move.controller import Controller

from mdp.model.tabular.tabular_model import TabularModel
from mdp.scenario.position_move.model.environment import Environment


class Model(TabularModel, ABC):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = self._controller
        self.environment: Optional[Environment] = self.environment
