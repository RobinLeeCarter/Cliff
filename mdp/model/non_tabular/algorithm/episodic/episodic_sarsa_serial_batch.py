from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_batch import NonTabularEpisodicBatch
from mdp.model.non_tabular.agent.non_tabular_episode import NonTabularEpisode
from mdp.model.non_tabular.value_function.state_action.linear_state_action_function import LinearStateActionFunction


class EpisodicSarsaSerialBatch(NonTabularEpisodicBatch,
                               algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_SERIAL_BATCH,
                               algorithm_name="Episodic Sarsa Serial Batch"):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._requires_q = True
        self._episodes: list[NonTabularEpisode] = []
        self._w_copy: Optional[np.ndarray] = None

    def start_episodes(self):
        self._episodes = []
        assert isinstance(self.Q, LinearStateActionFunction)
        self._w_copy = self.Q.w.copy()

    def _start_episode(self):
        self._agent.choose_action()

    def _do_training_step(self):
        ag = self._agent
        ag.take_action()
        ag.choose_action()

        target: float = ag.r + self._gamma * self.Q[ag.state, ag.action]
        delta: float = target - self.Q[ag.prev_state, ag.prev_action]
        alpha_delta: float = self._alpha * delta

        if self.Q.has_sparse_feature:
            gradient_indices: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
            self.Q.update_weights_sparse(indices=gradient_indices, delta_w=alpha_delta)
        else:
            gradient_vector: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
            delta_w: np.ndarray = alpha_delta * gradient_vector
            self.Q.update_weights(delta_w)

    def _end_episode(self):
        if self._agent.state.is_terminal:
            self._episodes.append(self._agent.episode)

    def apply_episodes(self):
        """for use with batch episodes but a single process"""
        # copy back original w and reapply episodes
        assert isinstance(self.Q, LinearStateActionFunction)
        self.Q.w = self._w_copy
        for episode in self._episodes:
            self._apply_episode(episode)

    def _apply_episode(self, episode: NonTabularEpisode):
        prev_state: Optional[NonTabularState] = None
        prev_action: Optional[NonTabularAction] = None
        reward: float
        state: Optional[NonTabularState]
        action: Optional[NonTabularAction]
        for reward_state_action in episode.trajectory:
            reward, state, action = reward_state_action.tuple
            if prev_state is not None:
                self._apply_sarsa(prev_state, prev_action, reward, state, action)
            prev_state, prev_action = state, action

    def _apply_sarsa(self,
                     prev_state: NonTabularState,
                     prev_action: NonTabularAction,
                     reward: float,
                     state: NonTabularState,
                     action: NonTabularAction):
        target: float = reward + self._gamma * self.Q[state, action]
        delta: float = target - self.Q[prev_state, prev_action]
        alpha_delta: float = self._alpha * delta

        if self.Q.has_sparse_feature:
            gradient_indices: np.ndarray = self.Q.get_gradient(prev_state, prev_action)
            self.Q.update_weights_sparse(indices=gradient_indices, delta_w=alpha_delta)
        else:
            gradient_vector: np.ndarray = self.Q.get_gradient(prev_state, prev_action)
            delta_w: np.ndarray = alpha_delta * gradient_vector
            self.Q.update_weights(delta_w)
