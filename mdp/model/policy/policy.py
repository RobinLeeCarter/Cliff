from __future__ import annotations
import abc
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment


class Policy(abc.ABC):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        self._environment = environment_
        self._policy_parameters: common.PolicyParameters = policy_parameters
        # TODO: maintain policy matrix

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
    def get_probability(self, state: State, action: Action) -> float:
        pass

    def get_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)
        for s, state in enumerate(self._environment.states):
            for a, action in enumerate(self._environment.actions_for_state(state)):
                probability = self.get_probability(state, action)
                policy_matrix[s, a] = probability
        return policy_matrix
