from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING, TypeVar, Generic

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment

from mdp.model.non_tabular.agent.reward_state_action import RewardStateAction, Trajectory
from mdp.model.non_tabular.agent.reward_feature_vector import RewardFeatureVector, FeatureTrajectory

from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.base.agent.base_episode import BaseEpisode

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class NonTabularEpisode(Generic[State, Action], BaseEpisode):
    """Just makes a record laid out in the standard way with Reward, State, Action for each _t"""
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 gamma: float,
                 step_callback: Optional[Callable[[], bool]] = None,
                 record_feature_trajectory: bool = False):
        super().__init__(environment, gamma, step_callback)
        self._environment: NonTabularEnvironment = environment
        # self.gamma: float = gamma
        # self._step_callback: Optional[Callable[[], bool]] = step_callback
        # self.record_first_visits = record_first_visits

        # R0=0, S0, A0, R1, S1, A1, R2 ... S(T-1), A(T-1), R(T), S(T), A(T)=None
        self._trajectory: Trajectory = []
        self.terminates: bool = False
        self.T: Optional[int] = None
        self.G: list[float] = []

        self.cont: bool = True

        self._record_feature_trajectory: bool = record_feature_trajectory
        self._feature_trajectory: FeatureTrajectory = []

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def feature_trajectory(self) -> FeatureTrajectory:
        return self._feature_trajectory

    @property
    def last_state(self) -> Optional[State]:
        if self._trajectory:
            return self._trajectory[-1].state
        else:
            return None

    @property
    def last_action(self) -> Optional[Action]:
        if self._trajectory:
            return self._trajectory[-1].action
        else:
            return None

    @property
    def prev_state(self) -> Optional[State]:
        if self._trajectory and len(self._trajectory) > 1:
            return self._trajectory[-2].state
        else:
            return None

    @property
    def prev_action(self) -> Optional[Action]:
        if self._trajectory and len(self._trajectory) > 1:
            return self._trajectory[-2].action
        else:
            return None

    # @profile
    def add_rsa(self,
                reward: float,
                state: State,
                action: Optional[Action],
                feature_vector: Optional[np.ndarray] = None):
        rsa = RewardStateAction(reward, state, action)
        self._trajectory.append(rsa)
        if self._record_feature_trajectory:
            reward_feature_vector = RewardFeatureVector(reward, feature_vector)
            self._feature_trajectory.append(reward_feature_vector)
        if state.is_terminal:
            self.terminates = True
            self.T = len(self._trajectory) - 1
        if self._step_callback:
            self.cont = self._step_callback()

    def generate_returns(self):
        if self.terminates:
            self.G = [0.0 for _ in range(self.T+1)]
            for t in range(self.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                self.G[t] = self[t+1].r + self.gamma * self.G[t + 1]

    def __getitem__(self, t: int) -> RewardStateAction:
        return self._trajectory[t]

    @property
    def max_t(self) -> int:
        if self._trajectory:
            return len(self._trajectory) - 1
        else:
            return 0

    @property
    def total_return(self) -> float:
        if self.G:
            return self.G[0]
        elif self.terminates:
            g: float = 0.0
            for t in range(self.T - 1, -1, -1):     # T-1, T-2, ... 1, 0
                g = self._trajectory[t + 1].r + self.gamma * g
            return g
        else:
            # not ideal but need to return something for some use-cases
            g: float = 0.0
            for t in range(len(self._trajectory) - 2, -1, -1):     # T-1, T-2, ... 1, 0
                g = self._trajectory[t + 1].r + self.gamma * g
            return g

        # g: float = 0
        # for t, rsa_ in enumerate(self.trajectory):
        #     if t > 0:
        #         g = rsa_.r + self.gamma * g
        # return g

    def get_state(self, t: int) -> State:
        return self._trajectory[t].state

    def get_action(self, t: int) -> Optional[Action]:
        return self._trajectory[t].action

    def get_s_a_g(self, t: int) -> tuple[State, Action, float]:
        rsa = self._trajectory[t]
        return rsa.state, rsa.action, self.G[t]
