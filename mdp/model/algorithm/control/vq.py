from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.episodic_online_control import EpisodicOnlineControl


class VQ(EpisodicOnlineControl):
    def __init__(self,
                 environment_: Environment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self._alpha_variable: bool = self._algorithm_parameters.alpha_variable
        self._alpha: float = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.VQ
        self.name = common.algorithm_name[self._algorithm_type]
        if self._alpha_variable:
            self.title = f"{self.name} α=0.5 then α=0.1"
        else:
            self.title = f"{self.name} α={self._alpha}"
        self._create_v()
        self._create_q()

    def parameter_changes(self, iteration: int):
        if self._alpha_variable:
            if iteration <= 50:
                self._alpha = 0.5
            else:
                self._alpha = 0.1

            # if iteration <= 20:
            #     self._alpha = 0.5
            # else:
            #     self._alpha = 10/iteration

    def _do_training_step(self):
        ag = self._agent
        ag.choose_action()
        ag.take_action()

        target = ag.r + self._gamma * self.V[ag.s]

        v_delta = target - self.V[ag.prev_s]
        self.V[ag.prev_s] += self._alpha * v_delta

        q_delta = target - self.Q[ag.prev_s, ag.prev_a]
        self.Q[ag.prev_s, ag.prev_a] += self._alpha * q_delta

        # update policy to be in-line with Q
        self._agent.policy[ag.prev_s] = self.Q.argmax[ag.prev_s]
