from __future__ import annotations
from typing import TypeVar, Generic, Optional
from abc import ABC

from mdp.model.base.base_model import BaseModel
from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp.model.tabular.agent.tabular_episode import TabularEpisode
from mdp.model.tabular.environment.tabular_state import TabularState
from mdp.model.tabular.environment.tabular_action import TabularAction

State = TypeVar('State', bound=TabularState)
Action = TypeVar('Action', bound=TabularAction)
Environment = TypeVar('Environment', bound=TabularEnvironment)


class TabularModel(Generic[State, Action, Environment],
                   BaseModel[Environment, TabularAgent[State, Action]],
                   ABC):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def _create_agent(self) -> TabularAgent[State, Action]:
        return TabularAgent[State, Action](self.environment)

    def _display_step(self, episode: Optional[TabularEpisode]):
        # if self.trainer.episode_counter >= 9900:
        self.update_grid_value_functions()
        self._controller.display_step(episode)

    def update_grid_value_functions(self):
        policy_for_display = self.target_policy.linked_policy
        self.environment.update_grid_value_functions(algorithm=self.agent.algorithm,
                                                     policy=policy_for_display)
