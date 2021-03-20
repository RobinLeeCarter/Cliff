from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.state import State


@dataclass(frozen=True)
class Outcome:
    state: State            # s'
    reward: float           # r
    probability: float      # p(s',r|s,a)
