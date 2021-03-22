from __future__ import annotations
from dataclasses import dataclass

from mdp.model import environment


@dataclass(frozen=True)
class State(environment.State):
    ending_cars_1: int
    ending_cars_2: int
