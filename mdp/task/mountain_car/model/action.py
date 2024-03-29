from __future__ import annotations
from dataclasses import dataclass

from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction


@dataclass(frozen=True)
class Action(NonTabularAction):
    """
    "There are three possible actions:
    full throttle forward (+1),
    full throttle reverse (-1),
    and zero throttle (0)"
    """
    acceleration: float

    def _get_categories(self) -> list:
        return [self.acceleration]
