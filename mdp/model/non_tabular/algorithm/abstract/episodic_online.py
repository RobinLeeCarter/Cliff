from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

from mdp.model.non_tabular.value_function.linear_state_action_function import LinearStateActionFunction

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp import common

from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm


class EpisodicOnline(NonTabularAlgorithm, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._alpha = self._algorithm_parameters.alpha
        self.Q: Optional[LinearStateActionFunction] = None

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return f"{name} α={algorithm_parameters.alpha}"

    def do_episode(self, episode_length_timeout: int):
        self._agent.start_episode()
        self._start_episode()
        while (not self._agent.state.is_terminal)\
                and self._agent.t < episode_length_timeout\
                and self._agent.episode.cont:
            self._do_training_step()

    def _start_episode(self):
        pass

    @abstractmethod
    def _do_training_step(self):
        pass

    def apply_result(self, result: common.Result):
        raise Exception("apply_result not implemented")
