from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction


@dataclass
class RSA:
    r: float
    state: NonTabularState
    action: Optional[NonTabularAction]

    @property
    def tuple(self) -> tuple:
        return self.r, self.state, self.action
