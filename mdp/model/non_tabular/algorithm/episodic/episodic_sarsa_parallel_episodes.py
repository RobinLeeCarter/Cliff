from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from mdp.model.non_tabular.agent.non_tabular_episode import NonTabularEpisode
from mdp.model.non_tabular.agent.reward_state_action import RewardStateAction
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_batch import NonTabularEpisodicBatch


class EpisodicSarsaParallelEpisodes(NonTabularEpisodicBatch,
                                    algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA_PARALLEL_EPISODES,
                                    algorithm_name="Episodic Sarsa Parallel Episodes"):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._requires_q = True

        self._previous_q: float = 0.0
        self._previous_gradient: Optional[np.ndarray] = None

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
            # self.Q.update_delta_weights_sparse(indices=self._previous_gradient, delta_w=alpha_delta)
            self.Q.update_weights_sparse(indices=self._previous_gradient, delta_w=alpha_delta)
        else:
            # gradient_vector: np.ndarray = self.Q.get_gradient(ag.prev_state, ag.prev_action)
            delta_w: np.ndarray = alpha_delta * self._previous_gradient
            # self.Q.update_delta_weights(delta_w)
            self.Q.update_weights(delta_w)

        if not ag.state.is_terminal:
            self._previous_q = current_q
            self._previous_gradient = self.Q.get_gradient(ag.state, ag.action)

    def _end_episode(self):
        if self._agent.state.is_terminal:
            self._episodes.append(self._agent.episode)

    def _apply_episode(self, episode: NonTabularEpisode):
        reward: float
        state: Optional[NonTabularState]
        action: Optional[NonTabularAction]

        reward_state_action: RewardStateAction = episode.trajectory[0]
        reward, state, action = reward_state_action.tuple
        self._previous_q = self.Q[state, action]
        self._previous_gradient = self.Q.get_gradient(state, action)

        for reward_state_action in episode.trajectory:
            reward, state, action = reward_state_action.tuple
            self._apply_sarsa(reward, state, action)

    def _apply_sarsa(self,
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
