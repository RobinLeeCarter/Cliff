from typing import List, Optional

import constants
import environment
from agent import rsa


class Episode:
    """Just makes a record laid out in the standard way with Reward, State, Action for each t"""
    def __init__(self):
        self.trajectory: List[rsa.RSA] = []
        self.rsa: rsa.RSA = rsa.RSA(None, None, None)

    def add_rsa(self,
                reward: Optional[float],
                state: Optional[environment.State],
                action: Optional[environment.Action]):
        self.rsa = rsa.RSA(reward, state, action)
        # S0, A0, R1, S1, A1, R2 ... S(T-1), A(T-1), R(T)
        self.trajectory.append(self.rsa)

    @property
    def max_t(self) -> int:
        if self.trajectory:
            return len(self.trajectory) - 1
        else:
            return 0

    @property
    def total_return(self) -> float:
        g: float = 0
        for t, rsa_ in enumerate(self.trajectory):
            if t > 0:
                g = rsa_.reward + constants.GAMMA * g
        return g
