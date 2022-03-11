from __future__ import annotations
from typing import TYPE_CHECKING, Type  # , TypeVar, Generic

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.agent.agent import Agent
from mdp.model.non_tabular.algorithm.abstract.algorithm import Algorithm
from mdp import common
# from mdp.model.tabular.environment.tabular_state import TabularState
# from mdp.model.tabular.environment.tabular_action import TabularAction


# State = TypeVar('State', bound=TabularState)
# Action = TypeVar('Action', bound=TabularAction)


class AlgorithmFactory:     # Generic[State, Action]
    def __init__(self, environment: NonTabularEnvironment, agent: Agent):      # [State, Action]
        self._environment: NonTabularEnvironment = environment                 # [State, Action]
        self._agent: Agent = agent

        self._algorithm_lookup: dict[common.NonTabularAlgorithmType, Type[Algorithm]] = self._get_algorithm_lookup()
        self._name_lookup: dict[common.NonTabularAlgorithmType, str] = self._get_name_lookup()

    def _get_algorithm_lookup(self) -> dict[common.NonTabularAlgorithmType, Type[Algorithm]]:
        # a = common.NonTabularAlgorithmType
        return {
        }

    def _get_name_lookup(self) -> dict[common.NonTabularAlgorithmType, str]:
        # a = common.NonTabularAlgorithmType
        return {
        }

    def create(self, algorithm_parameters: common.Settings.algorithm_parameters) -> Algorithm:
        algorithm_type: common.NonTabularAlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[Algorithm] = self._algorithm_lookup[algorithm_type]
        algorithm_name: str = self._name_lookup[algorithm_type]

        algorithm: Algorithm = type_of_algorithm(self._environment,
                                                 self._agent,
                                                 algorithm_parameters,
                                                 algorithm_name)
        return algorithm
