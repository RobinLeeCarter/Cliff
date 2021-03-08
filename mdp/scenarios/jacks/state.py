from __future__ import annotations
from dataclasses import dataclass

from mdp.model import environment


@dataclass(frozen=True)
class State(environment.State):
    # origin at bottom left
    cars_cob_1: int
    cars_cob_2: int
