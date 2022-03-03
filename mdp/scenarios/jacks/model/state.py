from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.general import state


@dataclass(frozen=True)
class State(state.State):
    ending_cars_1: int
    ending_cars_2: int
