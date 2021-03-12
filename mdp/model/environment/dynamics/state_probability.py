from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment import State


@dataclass(frozen=True)
class StateProbability:
    state: State
    probability: float
