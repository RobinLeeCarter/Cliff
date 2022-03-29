from __future__ import annotations
from typing import Optional, TYPE_CHECKING, TypeVar
from abc import ABC

if TYPE_CHECKING:
    from mdp.task._position_move.controller import Controller
from mdp.model.tabular.tabular_model import TabularModel
from mdp.task._position_move.model import environment
from mdp.task._position_move.model.state import State
from mdp.task._position_move.model.action import Action

Environment = TypeVar('Environment', bound=environment.Environment)


class Model(TabularModel[State, Action, Environment], ABC):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._controller: Optional[Controller] = None
