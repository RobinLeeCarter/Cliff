from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
    from mdp import common
    from mdp.model.base.policy.policy_factory import PolicyFactory
from mdp.model.base.policy.base_policy import BasePolicy


class BaseAlgorithm(ABC):
    def __init__(self,
                 agent: BaseAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        self._agent: BaseAgent = agent
        # self._environment: GeneralEnvironment = self._agent.environment
        self._algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self._verbose = self._algorithm_parameters.verbose

        self.name: str = name
        self.title: str = self.get_title(name, algorithm_parameters)

        self._gamma: float = self._agent.gamma
        self._target_policy: Optional[BasePolicy] = None
        self._behaviour_policy: Optional[BasePolicy] = None     # if on-policy = self._policy
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

    def __repr__(self):
        return f"{self.title}"

    @property
    def target_policy(self) -> Optional[BasePolicy]:
        return self._target_policy

    @property
    def behaviour_policy(self) -> Optional[BasePolicy]:
        return self._behaviour_policy

    def create_policies(self, policy_factory: PolicyFactory, settings: common.Settings):
        primary_policy = policy_factory.create(settings.policy_parameters)
        self._dual_policy_relationship = settings.dual_policy_relationship
        if self._dual_policy_relationship == common.DualPolicyRelationship.SINGLE_POLICY:
            self._target_policy = primary_policy
            self._behaviour_policy = primary_policy
        elif self._dual_policy_relationship == common.DualPolicyRelationship.INDEPENDENT_POLICIES:
            self._target_policy = primary_policy
            self._behaviour_policy = policy_factory.create(settings.behaviour_policy_parameters)
        elif self._dual_policy_relationship == common.DualPolicyRelationship.LINKED_POLICIES:
            self._target_policy = primary_policy.linked_policy     # typically the deterministic part we want to output
            self._behaviour_policy = primary_policy
        else:
            raise NotImplementedError

        self._agent.set_behaviour_policy(self._behaviour_policy)

    def initialize(self):
        pass

    def parameter_changes(self, iteration: int):
        pass

    # noinspection PyUnusedLocal
    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return name

    @abstractmethod
    def apply_result(self, result: common.Result):
        pass
