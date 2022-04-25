from __future__ import annotations
from typing import Optional

import numpy as np

from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp.model.non_tabular.algorithm.batch_mixin.batch_shared_weights import BatchSharedWeights
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa import Sarsa
from mdp import common


class SarsaSharedWeightsParallel(Sarsa, BatchSharedWeights,
                                 algorithm_type=common.AlgorithmType.NT_SARSA_SHARED_WEIGHTS_PARALLEL,
                                 algorithm_name="Sarsa Parallel Shared Weights"):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._previous_feature_vector: Optional[np.ndarray] = None

    def _start_episode(self):
        ag = self._agent
        ag.choose_action()
        self._previous_feature_vector = ag.episode[0].feature_vector
        # self._previous_q = self.Q[ag.state, ag.action]
        # self._previous_gradient = self.Q.get_gradient(ag.state, ag.action)

    def _do_training_step(self):
        ag = self._agent
        ag.take_action()
        ag.choose_action()
        current_feature_vector = ag.episode[ag.t].feature_vector
        previous_gradient = self.Q.calc_gradient(self._previous_feature_vector)

        with self._shared_w.lock:
            # a Transaction with consistent values of previous_q, current_q and weights to be updated
            previous_q = self.Q.calc_value(self._previous_feature_vector)
            if ag.state.is_terminal:
                current_q = 0.0
            else:
                current_q = self.Q.calc_value(current_feature_vector)

            # previous_q = self._previous_q
            # previous_gradient = self._previous_gradient
            # current_feature_vector = ag.episode[ag.t].feature_vector
            # print(f"{ag.state=} {ag.action=}")
            # if ag.state.is_terminal:
            #     current_q = 0.0
            # else:
            #     current_feature_vector = ag.episode[ag.t].feature_vector
            #     # current_feature_vector = self.Q.get_gradient(ag.state, ag.action)
            #     current_q = self.Q.calc_value(current_feature_vector)
            # current_q = self.Q[ag.state, ag.action]

            target: float = ag.r + self._gamma * current_q
            delta: float = target - previous_q
            alpha_delta: float = self._alpha * delta

            if self.Q.has_sparse_feature:
                # gradient_indices: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
                self.Q.update_weights_sparse(indices=previous_gradient, delta_w=alpha_delta)
            else:
                # gradient_vector: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
                delta_w: np.ndarray = alpha_delta * previous_gradient
                # TODO: fix so weights are unchanged
                self.Q.update_weights(delta_w)

        if not ag.state.is_terminal:
            self._previous_feature_vector = current_feature_vector
            # self._previous_q = current_q
            # self._previous_gradient = self.Q.get_gradient(ag.state, ag.action)
