from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.dynamics import state_probability
from mdp.scenarios.jacks.state import State


@dataclass(frozen=True)
class StateProbability(state_probability.StateProbability):
    state: State
