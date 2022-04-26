from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable, TypeVar, Generic

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.agent.non_tabular_episode import NonTabularEpisode
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp.model.base.agent.base_agent import BaseAgent
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class NonTabularAgent(Generic[State, Action], BaseAgent):
    def __init__(self, environment: NonTabularEnvironment[State, Action]):
        super().__init__(environment)
        self._environment: NonTabularEnvironment[State, Action] = environment
        self._behaviour_policy: Optional[NonTabularPolicy[State, Action]] = None
        self._episode: Optional[NonTabularEpisode[State, Action]] = None

        self.store_feature_vectors: bool = False
        self.store_feature_trajectories: bool = False
        self._feature: Optional[BaseFeature[State, Action]] = None
        self._feature_vector: Optional[np.ndarray] = None

        # always refers to values for time-step t
        self.r: float = 0.0
        self.state: Optional[State] = None
        self.action: Optional[Action] = None

        # always refers to values for time-step t-1
        self.prev_r: float = 0.0
        self.prev_state: Optional[State] = None
        self.prev_action: Optional[Action] = None

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    @property
    def environment(self) -> NonTabularEnvironment[State, Action]:
        return self._environment

    @property
    def episode(self) -> NonTabularEpisode:
        return self._episode

    def set_behaviour_policy(self, policy: NonTabularPolicy[State, Action]):
        self._behaviour_policy = policy

    def set_feature(self, feature: BaseFeature[State, Action]):
        self._feature = feature

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    # def generate_episodes(self,
    #                       num_episodes: int,
    #                       episode_length_timeout: Optional[int] = None
    #                       ) -> list[NonTabularEpisode]:
    #     return [self.generate_episode(episode_length_timeout)
    #             for _ in range(num_episodes)]

    def generate_episode(self,
                         episode_length_timeout: Optional[int] = None,
                         ) -> NonTabularEpisode:
        if not episode_length_timeout:
            episode_length_timeout = self._episode_length_timeout

        self.start_episode()
        while not self.state.is_terminal and self.t < episode_length_timeout:
            self.choose_action()
            if self._verbose:
                self._print_step()
            self.take_action()

        if self.t == episode_length_timeout:
            print("Warning: Failed to terminate")
        if self._verbose:
            print(f"t={self.t} \t state = {self.state} (terminal)")
        return self._episode

    def start_episode(self):
        """Gets initial state and sets initial reward to None"""
        if self._verbose:
            print("start episode...")
        self._episode = NonTabularEpisode(self._environment,
                                          self.gamma,
                                          self.store_feature_vectors,
                                          self.store_feature_trajectories,
                                          self._step_callback)
        self.t = 0
        self.state = self._environment.draw_start_state()
        self.action = None
        self.r = 0.0

    def choose_action(self):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        self.action = self._behaviour_policy[self.state]
        if self.store_feature_vectors or self.store_feature_trajectories:
            self._feature_vector = self._get_feature_vector()
        self._store_step()

    def _get_feature_vector(self) -> np.ndarray:
        feature_vector: np.ndarray
        if self._behaviour_policy.has_feature_vector and self._behaviour_policy.feature_vector is not None:
            feature_vector = self._behaviour_policy.feature_vector
        else:
            feature_vector = self._feature[self.state, self.action]
        return feature_vector

    def _store_step(self):
        self._episode.add_step(self.r, self.state, self.action, self._feature_vector)
        if self._verbose:
            print(f"t={self.t}\t{self.state} \t {self.action}")

    def take_action(self):
        """With state and action are already set,
        Perform action.
        Get new reward and state in response.
        Start a new time step with the new reward and state
        """
        new_r: float
        new_state: State
        new_r, new_state = self._environment.from_state_perform_action(self.state, self.action)

        # move time-step forward
        self.t += 1
        self.prev_r, self.prev_state, self.prev_action = self.r, self.state, self.action
        self.r, self.state, self.action = new_r, new_state, None

        if self.state.is_terminal:
            # add terminating step here as should not select another action
            self._store_step()

    def _print_step(self):
        print(f"t={self.t} \t state = {self.state} \t action = {self.action}")
