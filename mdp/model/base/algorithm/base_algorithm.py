from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Type
from abc import ABC

if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
    from mdp.factory.policy_factory import PolicyFactory
from mdp import common
from mdp.model.base.policy.base_policy import BasePolicy


class BaseAlgorithm(ABC):
    type_registry: dict[common.AlgorithmType, Type[BaseAlgorithm]] = {}
    name_registry: dict[common.AlgorithmType, str] = {}
    tabular: bool = False
    dynamic_programming: bool = False
    episodic: bool = False
    batch_episodes: Optional[common.BatchEpisodes] = None
    has_v: bool = False
    has_q: bool = False
    store_feature_vectors: bool = False
    store_feature_trajectories: bool = False

    def __init_subclass__(cls,
                          algorithm_type: Optional[common.AlgorithmType] = None,
                          algorithm_name: Optional[str] = None,
                          tabular: bool = False,
                          dynamic_programming: bool = False,
                          episodic: bool = False,
                          batch_episodes: Optional[common.BatchEpisodes] = None,
                          has_v: bool = False,
                          has_q: bool = False,
                          store_feature_vectors: bool = False,
                          store_feature_trajectories: bool = False,
                          **kwargs):
        super().__init_subclass__(**kwargs)
        if algorithm_type:
            BaseAlgorithm.type_registry[algorithm_type] = cls
            BaseAlgorithm.name_registry[algorithm_type] = algorithm_name
        if tabular:
            cls.tabular = tabular
        if dynamic_programming:
            cls.dynamic_programming = dynamic_programming
        if episodic:
            cls.episodic = episodic
        if batch_episodes:
            cls.batch_episodes = batch_episodes
        if has_v:
            cls.has_v = has_v
        if has_q:
            cls.has_q = has_q
        if store_feature_vectors:
            cls.store_feature_vectors = store_feature_vectors
        if store_feature_trajectories:
            cls.store_feature_trajectories = store_feature_trajectories

    def __init__(self,
                 agent: BaseAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        self._agent: BaseAgent = agent
        # self._environment: GeneralEnvironment = self._agent.environment
        self._algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self._verbose = self._algorithm_parameters.verbose

        self.name: str = BaseAlgorithm.name_registry[algorithm_parameters.algorithm_type]
        self.title: str = self.get_title(self.name, algorithm_parameters)

        self._gamma: float = self._agent.gamma
        self._target_policy: Optional[BasePolicy] = None
        self._behaviour_policy: Optional[BasePolicy] = None     # if on-policy = self._policy
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

    @staticmethod
    def get_title(name: str, algorithm_parameters: common.AlgorithmParameters) -> str:
        return name

    def __repr__(self):
        return f"{self.title}"

    @property
    def target_policy(self) -> Optional[BasePolicy]:
        return self._target_policy

    @property
    def behaviour_policy(self) -> Optional[BasePolicy]:
        return self._behaviour_policy

    def apply_settings(self, settings: common.Settings):
        self._agent.apply_settings(settings)

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

    def apply_results(self, results: list[common.Result]):
        raise Exception("apply_results not implemented")

    def apply_result(self, result: common.Result):
        raise Exception("apply_result not implemented")
