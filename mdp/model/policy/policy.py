from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment


class Policy(abc.ABC):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        self._environment = environment_
        self._policy_parameters: common.PolicyParameters = policy_parameters

    def __getitem__(self, state: State) -> Optional[Action]:
        if state.is_terminal:
            return None
        else:
            # this of course will go to the level in inheritance hierarchy set by self
            return self._get_action(state)

    def __setitem__(self, state: State, action: Action):
        raise NotImplementedError(f"__setitem__ not implemented for Policy: {type(self)}")

    @property
    def linked_policy(self) -> Policy:  # determinstic part if exists else self
        return self

    @abc.abstractmethod
    def _get_action(self, state: State) -> Optional[Action]:
        pass

    @abc.abstractmethod
    def get_probability(self, state_: State, action_: Action) -> float:
        pass
