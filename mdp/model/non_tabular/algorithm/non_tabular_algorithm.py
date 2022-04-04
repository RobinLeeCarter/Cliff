from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC

from mdp.model.non_tabular.algorithm.abstract.batch_episodes import BatchEpisodes
from mdp.model.non_tabular.feature.tile_coding.tile_coding import TileCoding

if TYPE_CHECKING:
    from mdp.model.non_tabular.value_function.state.state_function import StateFunction
    from mdp.model.non_tabular.value_function.state_action.state_action_function import StateActionFunction
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
    from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
    from mdp import common
from mdp.factory.feature_factory import FeatureFactory
from mdp.factory.value_function_factory import ValueFunctionFactory
from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm


class NonTabularAlgorithm(BaseAlgorithm, ABC):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._agent: NonTabularAgent = agent
        self._feature: Optional[BaseFeature] = None
        self._target_policy: Optional[NonTabularPolicy] = None
        self._behaviour_policy: Optional[NonTabularPolicy] = None     # if on-policy = self._policy
        self.V: Optional[StateFunction] = None
        self.Q: Optional[StateActionFunction] = None
        # self._environment: NonTabularEnvironment = self._agent.environment

        self._requires_v: bool = False
        self._requires_q: bool = False

    @property
    def target_policy(self) -> Optional[NonTabularPolicy]:
        return self._target_policy

    @property
    def behaviour_policy(self) -> Optional[NonTabularPolicy]:
        return self._behaviour_policy

    def create_feature_and_value_function(self,
                                          feature_factory: FeatureFactory,
                                          value_function_factory: ValueFunctionFactory,
                                          settings: common.Settings):
        if settings.feature_parameters:
            self._feature = feature_factory.create(settings.feature_parameters)
            self._update_parameters_based_on_feature()
            if isinstance(self._feature, TileCoding):   # issubclass(type(self), BatchEpisodes) and
                if self._feature.use_dict:
                    self._feature.build_complete_dict()

        if self._requires_v:
            self.V: StateFunction = value_function_factory.create_state_function(
                self._feature, settings.value_function_parameters
            )
        if self._requires_q:
            self.Q: StateActionFunction = value_function_factory.create_state_action_function(
                self._feature, settings.value_function_parameters
            )

        # link both policies to feature and q to cover all scenarios:
        # e.g. behaviour policy is random and off-policy case
        self._link_policy(self._behaviour_policy)
        self._link_policy(self._target_policy)

    def _update_parameters_based_on_feature(self):
        """e.g. update alpha based on number of tilings"""
        pass

    def _link_policy(self, policy: NonTabularPolicy):
        if policy.requires_feature:
            policy.set_feature(self._feature)
        if policy.requires_q:
            if self._requires_q:
                policy.set_state_action_function(self.Q)
            else:
                raise Exception("Policy requires Q but algorithm does not")
