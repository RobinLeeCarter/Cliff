from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.non_tabular import non_tabular_state


@dataclass(frozen=True)
class State(non_tabular_state.NonTabularState):
    position: float
    velocity: float

    def _get_floats(self) -> list[float]:
        return [self.position, self.velocity]
