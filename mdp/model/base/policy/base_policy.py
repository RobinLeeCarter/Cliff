from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.base.environment.base_environment import BaseEnvironment


class BasePolicy(ABC):
    def __init__(self, environment: BaseEnvironment, policy_parameters: common.PolicyParameters):
        self._environment: BaseEnvironment = environment
        self._policy_parameters: common.PolicyParameters = policy_parameters

    @property
    def linked_policy(self) -> BasePolicy:
        """Deterministic partner policy if exists else self"""
        return self
