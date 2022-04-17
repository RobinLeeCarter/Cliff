from __future__ import annotations

import numpy as np

from mdp.model.non_tabular.agent.reward_feature_vector import FeatureTrajectory
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp.model.non_tabular.algorithm.batch_mixin.batch_feature_trajectories import BatchFeatureTrajectories


class SarsaFeatureTrajectories(Sarsa, BatchFeatureTrajectories):
    def _apply_feature_trajectory(self, feature_trajectory: FeatureTrajectory):
        reward: float
        feature_vector: np.ndarray

        # starting state
        reward, feature_vector = feature_trajectory[0].tuple
        self._previous_q = self.Q.calc_value(feature_vector)
        self._previous_gradient = self.Q.calc_gradient(feature_vector)

        # body of trajectory (not first or last state)
        for reward_feature_vector in feature_trajectory[1:-1]:
            self._apply_reward_feature_vector(*reward_feature_vector.tuple)

        # terminal case
        self._apply_reward_feature_vector(*feature_trajectory[-1].tuple, is_terminal=True)

    def _apply_reward_feature_vector(self,
                                     reward: float,
                                     feature_vector: np.ndarray,
                                     is_terminal: bool = False):
        target: float
        current_q: float
        if is_terminal:
            current_q = 0.0
            target = reward
        else:
            current_q = self.Q.calc_value(feature_vector)
            target: float = reward + self._gamma * current_q
        delta: float = target - self._previous_q
        alpha_delta: float = self._alpha * delta

        if self.Q.has_sparse_feature:
            self.Q.update_weights_sparse(indices=self._previous_gradient, delta_w=alpha_delta)
        else:
            delta_w: np.ndarray = alpha_delta * self._previous_gradient
            self.Q.update_weights(delta_w)

        # check if last one?, perhaps doesn't matter
        if not is_terminal:
            self._previous_q = current_q
            self._previous_gradient = self.Q.calc_gradient(feature_vector)
