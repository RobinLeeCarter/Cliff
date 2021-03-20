from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.state import State


@dataclass(frozen=True)
class StateProbability:
    state: State            # s'
    probability: float      # p(s'|s,a)
