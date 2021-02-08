from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import environment


@dataclass
class RSA:
    reward: Optional[float]
    state: Optional[environment.State]
    action: Optional[environment.Action]

    @property
    def tuple(self) -> tuple:
        return self.reward, self.state, self.action
