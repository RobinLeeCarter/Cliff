from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_online import NonTabularEpisodicOnline


class EpisodicSarsa(NonTabularEpisodicOnline,
                    algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA,
                    algorithm_name="Episodic Sarsa"):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._requires_q = True

    def _start_episode(self):
        self._agent.choose_action()

    def apply_result(self, result: common.Result):
        raise Exception("apply_result not implemented")

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
            self.Q.update_weights(delta_w=delta_w)

        # self._target_policy[ag.prev_s] = self.Q.argmax[ag.prev_s]
        # print(f"a: {ag.a}"
        #       f"\tQ[curr]: {self.Q[ag.s, ag.a]}"
        #       f"\tQ[prev]: {self.Q[ag.prev_s, ag.prev_a]}"
        #       f"\tdelta: {delta}"
        #       f"\talpha: {self._alpha}")

        # previous verison: update policy to be in-line with Q by recalculation every time
        # ag.policy[ag.prev_s] = self.Q.argmax_over_actions(ag.prev_s)
