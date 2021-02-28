from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import math

if TYPE_CHECKING:
    from mdp.model import environment
from mdp import common
# renamed to avoid name conflicts
from mdp.model import algorithm as algorithm_
from mdp.model import policy as policy_
from mdp.model.agent import episode as episode_


class Agent:
    def __init__(self,
                 environment_: environment.Environment,
                 verbose: bool = False):
        self._environment: environment.Environment = environment_
        self._verbose: bool = verbose

        self._policy: Optional[policy_.Policy] = None
        self._algorithm: Optional[algorithm_.Episodic] = None
        self._episode: Optional[episode_.Episode] = None
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

        # always refers to values for time-step _t
        self.reward: Optional[float] = None
        self.state: Optional[environment.State] = None
        self.action: Optional[environment.Action] = None

        # always refers to values for time-step _t-1
        self.prev_reward: Optional[float] = None
        self.prev_state: Optional[environment.State] = None
        self.prev_action: Optional[environment.Action] = None

        self._response: Optional[environment.Response] = None

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    @property
    def policy(self) -> policy_.Policy:
        return self._policy

    @property
    def algorithm(self) -> algorithm_.Episodic:
        return self._algorithm

    @property
    def algorithm_title(self) -> str:
        return self._algorithm.title

    @property
    def episode(self) -> episode_.Episode:
        return self._episode

    def apply_settings(self, settings: common.Settings):
        self._policy = policy_.factory(self._environment, settings.policy_parameters)
        self._algorithm = algorithm_.factory(self._environment, self, settings.algorithm_parameters)
        self._episode_length_timeout = settings.episode_length_timeout
        self.gamma = settings.gamma

    def initialize(self):
        self._algorithm.initialize()

    def parameter_changes(self, iteration: int):
        # potentially change epsilon here
        self._algorithm.parameter_changes(iteration)

    def do_episode(self):
        self._algorithm.do_episode(self._episode_length_timeout)

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    def start_episode(self):
        """Gets initial state and sets initial reward to None"""
        if self._verbose:
            print("start episode...")
        self.t = 0
        self._episode = episode_.Episode(self.gamma, self._step_callback)

        # get starting state, reward will be None
        self._response = self._environment.start()
        self.reward = self._response.reward
        self.state = self._response.state

    def choose_action(self):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        self.action = self.policy[self.state]
        self._episode.add_rsa(reward=self.reward, state=self.state, action=self.action)
        if self._verbose:
            print(f"state = {self.state} \t action = {self.action}")

    def take_action(self):
        """With state and action are already set,
        Perform action.
        Get new reward and state in response.
        Start a new time step with the new reward and state
        """
        self._response = self._environment.from_state_perform_action(self.state, self.action)

        # move time-step forward
        self.t += 1
        self.prev_reward, self.prev_state, self.prev_action = self.reward, self.state, self.action
        self.reward, self.state, self.action = self._response.reward, self._response.state, None

        if self.state.is_terminal:
            # add terminating step here as should not select another action
            self._episode.add_rsa(reward=self.reward, state=self.state, action=self.action)

    def generate_episode(self, episode_length_timeout: Optional[int] = None) -> episode_.Episode:
        if not episode_length_timeout:
            episode_length_timeout = self._episode_length_timeout

        self.start_episode()
        while not self.state.is_terminal and self.t < episode_length_timeout:
            self.choose_action()
            if self._verbose:
                print(f"t={self.t} \t state = {self.state} \t action = {self.action}")
            self.take_action()
        if self.t == episode_length_timeout:
            print("Failed to terminate")
        if self._verbose:
            print(f"t={self.t} \t state = {self.state} (terminal)")
        return self._episode

    def print_statistics(self):
        self._algorithm.print_q_coverage_statistics()

    def rms_error(self) -> float:
        # better that it just fail if you use something with no V or an environment without get_optimum
        # if not self._algorithm.V or not hasattr(self._environment, 'get_optimum'):
        #     return None

        rms_error: float = 0.0
        for state in self._environment.states():
            if self._environment.is_valued_state(state):
                value: float = self._algorithm.V[state]
                # noinspection PyUnresolvedReferences
                optimum: float = self._environment.get_optimum(state)
                rms_error += (value - optimum)**2
        return rms_error
