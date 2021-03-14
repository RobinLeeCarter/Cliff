from __future__ import annotations
from dataclasses import dataclass

from mdp.scenarios.jacks.state import State


@dataclass(frozen=True)
class Outcome:
    state: State
    reward: float
    probability: float
