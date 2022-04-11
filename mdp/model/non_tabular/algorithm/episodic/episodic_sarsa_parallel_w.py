from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from mdp.model.non_tabular.agent.non_tabular_episode import NonTabularEpisode

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_batch import NonTabularEpisodicBatch


class EpisodicSarsaParallelW(NonTabularEpisodicBatch,
                             algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_PARALLEL_W,
                             algorithm_name="Episodic Sarsa Parallel W"):
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

    def _end_episode(self):
        pass

    def get_delta_weights(self) -> np.ndarray:
        return self.Q.get_delta_weights()

    def _apply_episode(self, episode: NonTabularEpisode):
        pass

    def apply_delta_w_vector(self, delta_w: np.ndarray):
        self.Q.update_weights(delta_w)
