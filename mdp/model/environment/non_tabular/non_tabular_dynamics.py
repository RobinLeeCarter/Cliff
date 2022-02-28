from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
from abc import ABC

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
from mdp.model.environment.non_tabular.non_tabular_action import NonTabularAction
from mdp.model.environment.dynamics import Dynamics


State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class NonTabularDynamics(Dynamics, Generic[State, Action], ABC):
    def __init__(self, environment: NonTabularEnvironment, environment_parameters: common.EnvironmentParameters):
        """init top down"""
        super().__init__(environment, environment_parameters)
        self._environment: NonTabularEnvironment = environment
        print("NonTabularDynamics init")

        # state_transition_probabilities[s',s,a] = p(s'|s,a)
        self.state_transition_probabilities: np.ndarray = np.array([], dtype=np.float)
        # expected_reward_np[s,a] = E[r|s,a] = Σs',r p(s',r|s,a).r
        self.expected_reward: np.ndarray = np.array([], dtype=np.float)
