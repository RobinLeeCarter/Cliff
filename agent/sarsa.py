from dataclasses import dataclass
from typing import Optional

import environment


@dataclass
class SARSA:
    prev_state: environment.State
    prev_action: environment.Action
    reward: float
    state: environment.State
    action: Optional[environment.Action]

    @property
    def tuple(self) -> tuple:
        return self.prev_state, self.prev_action, self.reward, self.state, self.action
