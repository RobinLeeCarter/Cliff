from __future__ import annotations
from typing import Type, TYPE_CHECKING

from mdp import common
if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm
from mdp.model.tabular.algorithm import tabular_algorithm_lookups
from mdp.model.non_tabular.algorithm import non_tabular_algorithm_lookups

from mdp.model.tabular.algorithm import tabular_algorithm_imports

if tabular_algorithm_imports.dummy_list:
    pass


class AlgorithmFactory:
    def __init__(self, agent: BaseAgent):
        self._agent: BaseAgent = agent

        self._algorithm_lookup: dict[common.AlgorithmType, Type[BaseAlgorithm]] = {}
        self._algorithm_lookup.update(tabular_algorithm_lookups.get_algorithm_lookup())
        self._algorithm_lookup.update(non_tabular_algorithm_lookups.get_algorithm_lookup())

        self._name_lookup: dict[common.AlgorithmType, str] = {}
        self._name_lookup.update(tabular_algorithm_lookups.get_name_lookup())
        self._name_lookup.update(non_tabular_algorithm_lookups.get_name_lookup())

    def create(self, algorithm_parameters: common.AlgorithmParameters) -> BaseAlgorithm:
        algorithm_type: common.AlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[BaseAlgorithm] = self._algorithm_lookup[algorithm_type]
        algorithm_name: str = self._name_lookup[algorithm_type]

        # TODO: not convinced that algorithm_name should be passed in since it's the same for all instances
        # class back to AlgorithmFactory with a static method? (circular?)
        algorithm: BaseAlgorithm = type_of_algorithm(self._agent,
                                                     algorithm_parameters,
                                                     algorithm_name)
        return algorithm

    def get_algorithm_name(self, algorithm_type: common.AlgorithmType) -> str:
        return self._name_lookup[algorithm_type]

    def get_algorithm_title(self, algorithm_parameters: common.AlgorithmParameters) -> str:
        algorithm_type: common.AlgorithmType = algorithm_parameters.algorithm_type
        type_of_algorithm: Type[BaseAlgorithm] = self._algorithm_lookup[algorithm_type]
        name: str = self._name_lookup[algorithm_type]
        return type_of_algorithm.get_title(name, algorithm_parameters)
