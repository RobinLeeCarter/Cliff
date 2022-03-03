from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.general import state


@dataclass(frozen=True)
class State(state.State):
    capital: int             # 0-100
