from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_batch import NonTabularEpisodicBatch


class EpisodicSarsaBatch(NonTabularEpisodicBatch,
                         algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_BATCH,
                         algorithm_name="Episodic Sarsa Batch"):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._requires_q = True

        self._previous_q: float = 0.0
        self._previous_gradient: Optional[np.ndarray] = None

    def start_episodes(self):
        self.Q.reset_delta_w()

    def _start_episode(self):
        ag = self._agent
        ag.choose_action()
        self._previous_q = self.Q[ag.state, ag.action]
        self._previous_gradient = self.Q.get_gradient(ag.state, ag.action)

    def _do_training_step(self):
        ag = self._agent
        ag.take_action()
        ag.choose_action()

        current_q = self.Q[ag.state, ag.action]
        target: float = ag.r + self._gamma * current_q
        delta: float = target - self._previous_q
        alpha_delta: float = self._alpha * delta

        if self.Q.has_sparse_feature:
            # gradient_indices: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
            self.Q.update_delta_weights_sparse(indices=self._previous_gradient, delta_w=alpha_delta)
            self.Q.update_weights_sparse(indices=self._previous_gradient, delta_w=alpha_delta)
        else:
            # gradient_vector: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
            delta_w: np.ndarray = alpha_delta * self._previous_gradient
            self.Q.update_delta_weights(delta_w)
            self.Q.update_weights(delta_w)

        if not ag.state.is_terminal:
            self._previous_q = current_q
            self._previous_gradient = self.Q.get_gradient(ag.state, ag.action)

    # end of process, multiprocessing
    def get_delta_weights(self) -> np.ndarray:
        """get the accumulated delta weights to pass out of the process in Result"""
        return self.Q.get_delta_weights()

    # end of batch, multiprocessing
    def apply_delta_w_vector(self, delta_w: np.ndarray):
        """in the parent process, apply the accumulated delta_w from all of the child processes"""
        self.Q.update_weights(delta_w)

    def apply_episodes(self):
        """for use with batch episodes but a single process"""
        self.Q.apply_delta_weights()
