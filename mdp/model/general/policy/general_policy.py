from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.general.environment.general_environment import GeneralEnvironment


class GeneralPolicy(ABC):
    def __init__(self, environment: GeneralEnvironment, policy_parameters: common.PolicyParameters):
        self._environment: GeneralEnvironment = environment
        self._policy_parameters: common.PolicyParameters = policy_parameters

    @property
    def linked_policy(self) -> GeneralPolicy:
        """Deterministic partner policy if exists else self"""
        return self
