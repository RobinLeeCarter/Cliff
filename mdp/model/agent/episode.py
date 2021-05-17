from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.action import Action
    from mdp.model.environment.state import State
    from mdp.model.environment.environment import Environment

from mdp.model.agent import rsa


class Episode:
    """Just makes a record laid out in the standard way with Reward, State, Action for each _t"""
    def __init__(self,
                 environment: Environment,
                 gamma: float,
                 step_callback: Optional[Callable[[], bool]] = None,
                 record_first_visits: bool = False):
        self._environment = environment
        self.gamma: float = gamma
        # TODO: move step_callback up to agent or even to algorithm
        self._step_callback: Optional[Callable[[], bool]] = step_callback
        self.record_first_visits = record_first_visits

        # S0, A0, R1, S1, A1, R2 ... S(T-1), A(T-1), R(T)
        self.trajectory: list[rsa.RSA] = []
        self.terminates: bool = False
        self.T: Optional[int] = None
        self.G: np.ndarray = np.array([], dtype=float)

        self.cont: bool = True

        if self.record_first_visits:
            # self.visited_s: np.ndarray = np.zeros(shape=len(self._environment.states), dtype=bool)
            self.is_first_visit: list[bool] = []
            self.visited_s: set[int] = set()

    @property
    def last_state(self) -> Optional[State]:
        if self.trajectory:
            last_s = self.trajectory[-1].s
            # if last_s is None:
            #     return None
            # else:
            return self._environment.states[last_s]
        else:
            return None

    @property
    def last_action(self) -> Optional[Action]:
        if self.trajectory:
            last_a: int = self.trajectory[-1].a
            if last_a == -1:
                return None
            else:
                return self._environment.actions[last_a]
        else:
            return None

    @property
    def prev_state(self) -> Optional[State]:
        if self.trajectory and len(self.trajectory) > 1:
            prev_s = self.trajectory[-2].s
            # if prev_s is None:
            #     return None
            # else:
            return self._environment.states[prev_s]
        else:
            return None

    @property
    def prev_action(self) -> Optional[Action]:
        if self.trajectory and len(self.trajectory) > 1:
            prev_a = self.trajectory[-2].a
            if prev_a == -1:
                return None
            else:
                return self._environment.actions[prev_a]
        else:
            return None

    # @profile
    def add_rsa(self,
                r: float,
                s: int,
                a: int,
                is_terminal: bool):
        rsa_ = rsa.RSA(r, s, a)
        self.trajectory.append(rsa_)
        if is_terminal:
            self.terminates = True
            self.T = len(self.trajectory) - 1
        if self.record_first_visits:
            self._first_visit_check(s)
        if self._step_callback:
            self.cont = self._step_callback()

    def generate_returns(self):
        if self.terminates:
            self.G = np.zeros(shape=self.T+1, dtype=float)
            self.G[self.T] = 0.0
            for t in range(self.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                self.G[t] = self[t+1].r + self.gamma * self.G[t + 1]

    def _first_visit_check(self, s: int):
        is_first_visit = (s not in self.visited_s)
        self.is_first_visit.append(is_first_visit)
        if is_first_visit:
            self.visited_s.add(s)

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
                g = rsa_.r + self.gamma * g
        return g

    def get_state(self, t: int) -> State:
        s: int = self.trajectory[t].s
        state: State = self._environment.states[s]
        return state

    def get_action(self, t: int) -> Optional[Action]:
        a: int = self.trajectory[t].a
        if a == -1:
            return None
        else:
            return self._environment.actions[a]
