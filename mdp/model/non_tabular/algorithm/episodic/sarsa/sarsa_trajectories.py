from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np

from mdp.model.non_tabular.agent.reward_state_action import RewardStateAction, Trajectory
from mdp.model.non_tabular.algorithm.batch_mixin.batch_trajectories import BatchTrajectories
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState


class SarsaTrajectories(Sarsa, BatchTrajectories, ABC):
    def _apply_trajectory(self, trajectory: Trajectory):
        reward: float
        state: Optional[NonTabularState]
        action: Optional[NonTabularAction]

        reward_state_action: RewardStateAction = trajectory[0]
        reward, state, action = reward_state_action.tuple
        self._previous_q = self.Q[state, action]
        self._previous_gradient = self.Q.get_gradient(state, action)

        for reward_state_action in trajectory[1:]:
            self._apply_rsa(*reward_state_action.tuple)

    def _apply_rsa(self,
                   reward: float,
                   state: NonTabularState,
                   action: NonTabularAction):
        current_q = self.Q[state, action]
        target: float = reward + self._gamma * current_q
        delta: float = target - self._previous_q
        alpha_delta: float = self._alpha * delta

        if self.Q.has_sparse_feature:
            # gradient_indices: np.ndarray = self.Q.get_gradient(prev_state, prev_action)
            self.Q.update_weights_sparse(indices=self._previous_gradient, delta_w=alpha_delta)
        else:
            # gradient_vector: np.ndarray = self.Q.get_gradient(prev_state, prev_action)
            delta_w: np.ndarray = alpha_delta * self._previous_gradient
            self.Q.update_weights(delta_w)

        if not state.is_terminal:
            self._previous_q = current_q
            self._previous_gradient = self.Q.get_gradient(state, action)
