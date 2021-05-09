from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.episodic_online import EpisodicOnline


class TD0(EpisodicOnline):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._alpha = self._algorithm_parameters.alpha
        self._algorithm_type = common.AlgorithmType.TD_0
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} α={self._alpha}"
        self._create_v()

    def _do_training_step(self):
        self._agent.choose_action()
        self._agent.take_action()

        prev_state = self._agent.prev_state
        reward = self._agent.reward
        state = self._agent.state

        target = reward + self._gamma * self.V[state]
        delta = target - self.V[prev_state]
        self.V[prev_state] += self._alpha * delta
