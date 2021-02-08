from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

import constants
from agent import rsa
if TYPE_CHECKING:
    import environment


class Episode:
    """Just makes a record laid out in the standard way with Reward, State, Action for each t"""
    def __init__(self):
        # S0, A0, R1, S1, A1, R2 ... S(T-1), A(T-1), R(T)
        self.trajectory: list[rsa.RSA] = []
        self.terminates: bool = False
        self.T: Optional[int] = None
        self.G: np.ndarray = np.array([], dtype=float)

    def add_rsa(self,
                reward: Optional[float],
                state: environment.State,
                action: Optional[environment.Action]):
        rsa_ = rsa.RSA(reward, state, action)
        self.trajectory.append(rsa_)

        if state.is_terminal:
            self.terminates = True
            self.T = len(self.trajectory) - 1

    def generate_returns(self):
        if self.terminates:
            self.G = np.zeros(shape=self.T+1, dtype=float)
            self.G[self.T] = 0.0
            for t in range(self.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                self.G[t] = self[t+1].reward + constants.GAMMA * self.G[t+1]

    def __getitem__(self, t: int) -> rsa.RSA:
        return self.trajectory[t]

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
