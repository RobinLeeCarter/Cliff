from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC

if TYPE_CHECKING:
    from mdp.model.non_tabular.value_function.state_function import StateFunction
    from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
    from mdp import common
from mdp.model.non_tabular.feature.feature_factory import FeatureFactory
from mdp.model.non_tabular.feature.feature import Feature
from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm


class NonTabularAlgorithm(BaseAlgorithm, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._agent: NonTabularAgent = agent
        self._feature: Optional[Feature] = None
        self.V: Optional[StateFunction] = None
        self.Q: Optional[StateActionFunction] = None
        # self._environment: NonTabularEnvironment = self._agent.environment

    @property
    def target_policy(self) -> Optional[NonTabularPolicy]:
        return self._target_policy

    @property
    def behaviour_policy(self) -> Optional[NonTabularPolicy]:
        return self._behaviour_policy

    def create_feature(self, feature_factory: FeatureFactory, feature_parameters: common.FeatureParameters):
        self._feature: Feature = feature_factory.create(feature_parameters)

    def apply_result(self, result: common.Result):
        raise Exception("apply_result not implemented")
