from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction


@dataclass(frozen=True)
class Action(NonTabularAction):
    """
    "There are three possible actions:
    full throttle forward (+1),
    full throttle reverse (-1),
    and zero throttle (0)"
    """
    acceleration: float

    def _get_discrete_tuple(self) -> tuple:
        return self.acceleration,
