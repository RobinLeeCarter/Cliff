from dataclasses import dataclass
from typing import Optional

import environment


@dataclass
class Sarsa:
    state: environment.State
    action: environment.Action
    reward: float
    next_state: environment.State
    next_action: Optional[environment.Action]

    @property
    def tuple(self) -> tuple:
        return self.state, self.action, self.reward, self.next_state, self.next_action
