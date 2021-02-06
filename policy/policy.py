import abc
from typing import Optional

import environment


class Policy(abc.ABC):
    def __init__(self, environment_: environment.Environment):
        self.environment = environment_

    def __getitem__(self, state: environment.State) -> Optional[environment.Action]:
        if state.is_terminal:
            return None
        else:
            return self.get_action(state)

    def __setitem__(self, state: environment.State, action: environment.Action):
        raise NotImplementedError(f"__setitem__ not implemented for Policy: {type(self)}")

    @abc.abstractmethod
    def get_action(self, state: environment.State) -> environment.Action:
        pass

    @abc.abstractmethod
    def get_probability(self, state_: environment.State, action_: environment.Action) -> float:
        pass
