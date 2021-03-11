from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from mdp.model import environment, agent
    from mdp import common
from mdp.model.algorithm.abstract import dynamic_programming


class DynamicProgrammingV(dynamic_programming.DynamicProgramming, abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._create_v()

    def _get_expected_return(self, state: environment.State, action: environment.Action) -> float:
        expected_return: float = self._dynamics.get_expected_reward(state, action)
        for state_probability in self._dynamics.get_next_state_distribution(state, action):
            expected_return += state_probability.probability * self._agent.gamma * self.V[state_probability.state]
        return expected_return
