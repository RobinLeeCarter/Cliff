from __future__ import annotations
from typing import Optional, Callable, TypeVar, Generic

from mdp import common
from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
from mdp.model.tabular.agent.tabular_episode import TabularEpisode
from mdp.model.tabular.policy.tabular_policy import TabularPolicy
from mdp.model.base.agent.base_agent import BaseAgent
from mdp.model.tabular.environment.tabular_state import TabularState
from mdp.model.tabular.environment.tabular_action import TabularAction

State = TypeVar('State', bound=TabularState)
Action = TypeVar('Action', bound=TabularAction)


class TabularAgent(Generic[State, Action], BaseAgent):
    def __init__(self,
                 environment: TabularEnvironment[State, Action]):
        super().__init__(environment)
        self._environment: TabularEnvironment[State, Action] = environment

        self._behaviour_policy: Optional[TabularPolicy] = None

        self._episode: Optional[TabularEpisode] = None
        self._first_visit: bool = False

        # always refers to values for time-step t
        self.r: float = 0.0
        self.s: int = -1
        self.a: int = -1
        self.is_terminal: bool = False  # stored for performance

        # always refers to values for time-step t-1
        self.prev_r: float = 0.0
        self.prev_s: int = -1
        self.prev_a: int = -1

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    @property
    def environment(self) -> TabularEnvironment[State, Action]:
        return self._environment

    @property
    def episode(self) -> TabularEpisode:
        return self._episode

    def set_behaviour_policy(self, policy: TabularPolicy):
        self._behaviour_policy = policy

    def apply_settings(self, settings: common.Settings):
        super().apply_settings(settings)
        self._first_visit: bool = settings.algorithm_parameters.first_visit

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    # def generate_episodes(self,
    #                       num_episodes: int,
    #                       episode_length_timeout: Optional[int] = None,
    #                       exploring_starts: bool = False,
    #                       ) -> list[TabularEpisode]:
    #     return [self.generate_episode(episode_length_timeout, exploring_starts)
    #             for _ in range(num_episodes)]

    def generate_episode(self,
                         episode_length_timeout: Optional[int] = None,
                         exploring_starts: bool = False
                         ) -> TabularEpisode:
        if not episode_length_timeout:
            episode_length_timeout = self._episode_length_timeout

        self.start_episode(exploring_starts)
        while not self.is_terminal and self.t < episode_length_timeout:
            self.choose_action()
            if self._verbose:
                self._print_step()
            self.take_action()

        if self.t == episode_length_timeout:
            print("Warning: Failed to terminate")
        if self._verbose:
            state: TabularState = self._environment.states[self.s]
            print(f"t={self.t} \t state = {state} (terminal)")
        return self._episode

    def start_episode(self, exploring_starts: bool = False):
        """Gets initial state and sets initial reward to None"""
        env = self._environment

        if self._verbose:
            print("start episode...")
        self.t = 0

        self._episode = TabularEpisode(env, self.gamma, self._step_callback, self._first_visit)

        if exploring_starts:
            # completely random starting state and action and take the action, reward will be None
            # state, action = self._environment.get_random_state_action()
            self.s, self.a = env.s_a_distribution.draw_one()
            self.is_terminal = env.is_terminal[self.s]
            self.r = 0.0
            # action = self._environment.dynamics.get_random_action_for_state(self.state)
            self.assign_action(self.a)
            self.take_action()
        else:
            # get starting state, reward will be None
            self.s = env.start_s_distribution.draw_one()
            # self.s = env.start_s()
            self.is_terminal = env.is_terminal[self.s]
            self.r = 0.0

    def choose_action(self):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        self.a = self._behaviour_policy[self.s]
        self._store_rsa()

    def assign_action(self, a: int):
        self.a = a
        self._store_rsa()

    def _store_rsa(self):
        # is_terminal = self._environment.states[self.s].is_terminal
        self._episode.add_rsa(self.r, self.s, self.a, self.is_terminal)
        if self._verbose:
            state: TabularState = self._environment.states[self.s]
            action: Optional[TabularAction]
            if self.a == -1:
                action = None
            else:
                action = self._environment.actions[self.a]
            print(f"state = {state} \t action = {action}")

    def take_action(self):
        """With state and action are already set,
        Perform action.
        Get new reward and state in response.
        Start a new time step with the new reward and state
        """
        new_r, new_s, self.is_terminal = self._environment.from_s_perform_a(self.s, self.a)

        # move time-step forward
        self.t += 1
        self.prev_r, self.prev_s, self.prev_a = self.r, self.s, self.a
        self.r, self.s, self.a = new_r, new_s, -1

        if self.is_terminal:
            # add terminating step here as should not select another action
            self._episode.add_rsa(self.r, self.s, self.a, self.is_terminal)

    def _print_step(self):
        state: TabularState = self._environment.states[self.s]
        action: Optional[TabularAction]
        if self.a == -1:
            action = None
        else:
            action = self._environment.actions[self.a]
        print(f"t={self.t} \t state = {state} \t action = {action}")
