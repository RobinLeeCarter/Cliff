from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action


@dataclass
class RSA:
    reward: Optional[float]
    state: Optional[State]
    action: Optional[Action]

    @property
    def tuple(self) -> tuple:
        return self.reward, self.state, self.action
