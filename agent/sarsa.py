from dataclasses import dataclass
from typing import Optional

import environment


@dataclass
class SARSA:
    state: environment.State
    action: environment.Action
    next_reward: float
    next_state: environment.State
    next_action: Optional[environment.Action]

    @property
    def tuple(self) -> tuple:
        return self.state, self.action, self.next_reward, self.next_state, self.next_action
