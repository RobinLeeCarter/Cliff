from __future__ import annotations
from dataclasses import dataclass

from mdp.model.tabular.environment.tabular_state import TabularState


@dataclass(frozen=True)
class State(TabularState):
    ending_cars_1: int
    ending_cars_2: int
