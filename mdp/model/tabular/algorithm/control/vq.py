from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_online_control import EpisodicOnlineControl


class VQ(EpisodicOnlineControl):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._alpha_variable: bool = self._algorithm_parameters.alpha_variable
        self._alpha: float = self._algorithm_parameters.alpha
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
        self._target_policy[ag.prev_s] = self.Q.argmax[ag.prev_s]

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        if algorithm_parameters.alpha_variable:
            return f"{name} α=0.5 then α=0.1"
        else:
            return f"{name} α={algorithm_parameters.alpha}"
