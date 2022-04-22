from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Type, Optional

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.base.environment.base_environment import BaseEnvironment


class BasePolicy(ABC):
    type_registry: dict[common.PolicyType, Type[BasePolicy]] = {}
    name_registry: dict[common.PolicyType, str] = {}
    requires_q: bool = False
    requires_feature: bool = False
    has_feature_vector: bool = False

    def __init_subclass__(cls,
                          policy_type: Optional[common.PolicyType] = None,
                          policy_name: str = None,
                          requires_q: bool = False,
                          requires_feature: bool = False,
                          has_feature_vector: bool = False,
                          **kwargs):
        super().__init_subclass__(**kwargs)
        if policy_type:
            BasePolicy.type_registry[policy_type] = cls
            BasePolicy.name_registry[policy_type] = policy_name
        if requires_q:
            cls.requires_q = requires_q
        if requires_feature:
            cls.requires_feature = requires_feature
        if has_feature_vector:
            cls.has_feature_vector = has_feature_vector

    def __init__(self, environment: BaseEnvironment, policy_parameters: common.PolicyParameters):
        self._environment: BaseEnvironment = environment
        self._policy_parameters: common.PolicyParameters = policy_parameters

    @property
    def linked_policy(self) -> BasePolicy:
        """Deterministic partner policy if exists else self"""
        return self
