from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC

from mdp.model.general.general_model import GeneralModel
from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
from mdp.model.tabular.agent.agent import Agent
from mdp.model.tabular.environment.tabular_state import TabularState
from mdp.model.tabular.environment.tabular_action import TabularAction

State = TypeVar('State', bound=TabularState)
Action = TypeVar('Action', bound=TabularAction)
Environment = TypeVar('Environment', bound=TabularEnvironment)


class TabularModel(Generic[State, Action, Environment],
                   GeneralModel[Environment, Agent[State, Action]],
                   ABC):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def _create_agent(self) -> Agent[State, Action]:
        return Agent[State, Action](self.environment)
