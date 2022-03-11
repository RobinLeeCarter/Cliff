from __future__ import annotations
from typing import Type, TYPE_CHECKING

from mdp import common
if TYPE_CHECKING:
    # from mdp.model.general.environment.general_environment import GeneralEnvironment
    from mdp.model.general.agent.general_agent import GeneralAgent
from mdp.model.general.algorithm.general_algorithm import GeneralAlgorithm
from mdp.model.tabular.algorithm import tabular_algorithm_lookups
from mdp.model.non_tabular.algorithm import non_tabular_algorithm_lookups


class AlgorithmFactory:
    def __init__(self, agent: GeneralAgent):
        self._agent: GeneralAgent = agent

        self._algorithm_lookup: dict[common.AlgorithmType, Type[GeneralAlgorithm]] = {}
        self._algorithm_lookup.update(tabular_algorithm_lookups.get_algorithm_lookup())
        self._algorithm_lookup.update(non_tabular_algorithm_lookups.get_algorithm_lookup())

        self._name_lookup: dict[common.AlgorithmType, str] = {}
        self._name_lookup.update(tabular_algorithm_lookups.get_name_lookup())
        self._name_lookup.update(non_tabular_algorithm_lookups.get_name_lookup())

    def create(self, algorithm_parameters: common.Settings.algorithm_parameters) -> GeneralAlgorithm:
        algorithm_type: common.AlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[GeneralAlgorithm] = self._algorithm_lookup[algorithm_type]
        algorithm_name: str = self._name_lookup[algorithm_type]

        algorithm: GeneralAlgorithm = type_of_algorithm(self._agent,
                                                        algorithm_parameters,
                                                        algorithm_name)
        return algorithm

    def lookup_algorithm_name(self, algorithm_type: common.AlgorithmType) -> str:
        return self._name_lookup[algorithm_type]
