from __future__ import annotations
from typing import TYPE_CHECKING    # , Optional
from abc import ABC

if TYPE_CHECKING:
    # from mdp.model.general.environment.general_environment import GeneralEnvironment
    from mdp.model.general.agent.general_agent import GeneralAgent
    # from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
    from mdp import common
# from mdp.model.tabular.algorithm import linear_algebra as la
# from mdp.model.non_tabular.value_function.state_function import StateFunction
# from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction


class GeneralAlgorithm(ABC):
    def __init__(self,
                 agent: GeneralAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        self._agent: GeneralAgent = agent
        # self._environment: GeneralEnvironment = self._agent.environment
        self._algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self._verbose = self._algorithm_parameters.verbose

        self.name: str = name
        self.title: str = name

        self._gamma: float = self._agent.gamma

    def __repr__(self):
        return f"{self.title}"

    def initialize(self):
        pass

    def parameter_changes(self, iteration: int):
        pass
