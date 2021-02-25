from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mdp.model import environment
from mdp.model.agent import rsa


class Episode:
    """Just makes a record laid out in the standard way with Reward, State, Action for each _t"""
    def __init__(self,
                 gamma: float,
                 step_callback: Optional[Callable[[], bool]] = None):
        self.gamma: float = gamma
        # S0, A0, R1, S1, A1, R2 ... S(T-1), A(T-1), R(T)
        self.trajectory: list[rsa.RSA] = []
        self.terminates: bool = False
        self.T: Optional[int] = None
        self.G: np.ndarray = np.array([], dtype=float)

        self._step_callback: Optional[Callable[[], bool]] = step_callback
        self.cont: bool = True

    @property
    def last_state(self) -> Optional[environment.State]:
        if self.trajectory:
            return self.trajectory[-1].state
        else:
            return None

    @property
    def last_action(self) -> Optional[environment.Action]:
        if self.trajectory:
            return self.trajectory[-1].action
        else:
            return None

    def add_rsa(self,
                reward: Optional[float],
                state: environment.State,
                action: Optional[environment.Action]):
        rsa_ = rsa.RSA(reward, state, action)
        self.trajectory.append(rsa_)
        if state.is_terminal:
            self.terminates = True
            self.T = len(self.trajectory) - 1
        if self._step_callback:
            self.cont = self._step_callback()

    def generate_returns(self):
        if self.terminates:
            self.G = np.zeros(shape=self.T+1, dtype=float)
            self.G[self.T] = 0.0
            for t in range(self.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                self.G[t] = self[t+1].reward + self.gamma * self.G[t + 1]

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
                g = rsa_.reward + self.gamma * g
        return g