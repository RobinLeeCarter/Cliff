from __future__ import annotations

import numpy as np

from mdp.model.non_tabular.agent.reward_feature_vector import RewardFeatureVector, FeatureTrajectory
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp.model.non_tabular.algorithm.batch_mixin.batch_feature_trajectories import BatchFeatureTrajectories
from mdp import common


class SarsaParallelFeatureTrajectories(Sarsa, BatchFeatureTrajectories,
                                       algorithm_type=common.AlgorithmType.NT_SARSA_PARALLEL_FEATURE_TRAJECTORIES,
                                       algorithm_name="Sarsa Parallel Feature Trajectories"):
    def _apply_feature_trajectory(self, feature_trajectory: FeatureTrajectory):
        reward: float
        feature_vector: np.ndarray

        reward_feature_vector: RewardFeatureVector = feature_trajectory[0]
        reward, feature_vector = reward_feature_vector.tuple
        self._previous_q = self.Q.get_feature_vector_value[feature_vector]
        self._previous_gradient = self.Q.get_feature_vector_gradient(feature_vector)

        for reward_feature_vector in feature_trajectory[1:]:
            self._apply_reward_feature_vector(*reward_feature_vector.tuple)

    def _apply_reward_feature_vector(self,
                                     reward: float,
                                     feature_vector: np.ndarray):
        current_q = self.Q.get_feature_vector_value(feature_vector)
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

        # check if last one, perhaps doesn't matter
        self._previous_q = current_q
        self._previous_gradient = self.Q.get_feature_vector_gradient(feature_vector)
