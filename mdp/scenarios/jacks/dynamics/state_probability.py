from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.dynamics import unused_state_probability
from mdp.scenarios.jacks.state import State


@dataclass(frozen=True)
class StateProbability(unused_state_probability.StateProbability):
    state: State
