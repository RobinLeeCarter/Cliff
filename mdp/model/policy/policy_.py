from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import common
    from mdp.model import environment


class Policy(abc.ABC):
    def __init__(self, environment_: environment.Environment, policy_parameters: common.PolicyParameters):
        self._environment = environment_
        self._policy_parameters: common.PolicyParameters = policy_parameters

    def __getitem__(self, state: environment.State) -> Optional[environment.Action]:
        if state.is_terminal:
            return None
        else:
            # this of course will go to the level in inheritance hierarchy set by self
            return self.get_action(state)

    def __setitem__(self, state: environment.State, action: environment.Action):
        raise NotImplementedError(f"__setitem__ not implemented for Policy: {type(self)}")

    @abc.abstractmethod
    def get_action(self, state: environment.State) -> environment.Action:
        pass

    @abc.abstractmethod
    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        pass
