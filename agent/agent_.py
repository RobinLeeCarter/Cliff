from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import environment
    import policy
from agent import episode


class Agent:
    def __init__(self,
                 environment_: environment.Environment,
                 policy_: policy.Policy,
                 verbose: bool = False):
        self.environment: environment.Environment = environment_
        self.policy: policy.Policy = policy_
        self.verbose: bool = verbose

        self.gamma: float = 1.0
        self.episode: Optional[episode.Episode] = None

        self.t: int = 0

        # always refers to values for time-step t
        self.reward: Optional[float] = None
        self.state: Optional[environment.State] = None
        self.action: Optional[environment.Action] = None

        # always refers to values for time-step t-1
        self.prev_reward: Optional[float] = None
        self.prev_state: Optional[environment.State] = None
        self.prev_action: Optional[environment.Action] = None

        self.response: Optional[environment.Response] = None

        # trainer callback
        self._step_callback: Optional[Callable[[episode.Episode], None]] = None

    def set_policy(self, policy_: policy.Policy):
        self.policy = policy_

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def set_step_callback(self, step_callback: Optional[Callable[[episode.Episode], None]] = None):
        self._step_callback = step_callback

    def start_episode(self):
        """Gets initial state and sets initial reward to None"""
        if self.verbose:
            print("start episode...")
        self.t = 0
        self.episode = episode.Episode(self.gamma)
        if self._step_callback:
            self.episode.set_step_callback(self._step_callback)

        # get starting state, reward will be None
        self.response = self.environment.start()
        self.reward = self.response.reward
        self.state = self.response.state

    def choose_action(self):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        self.action = self.policy[self.state]
        self.episode.add_rsa(reward=self.reward, state=self.state, action=self.action)
        if self.verbose:
            print(f"state = {self.state} \t action = {self.action}")

    def take_action(self):
        """With state and action are already set,
        Perform action.
        Get new reward and state in response.
        Start a new time step with the new reward and state
        """
        self.response = self.environment.from_state_perform_action(self.state, self.action)

        # move time-step forward
        self.t += 1
        self.prev_reward, self.prev_state, self.prev_action = self.reward, self.state, self.action
        self.reward, self.state, self.action = self.response.reward, self.response.state, None

        if self.state.is_terminal:
            # add terminating step here as should not select another action
            self.episode.add_rsa(reward=self.reward, state=self.state, action=self.action)

    def generate_episode(self, episode_length_timeout: int):
        self.start_episode()
        while not self.state.is_terminal and self.t < episode_length_timeout:
            self.choose_action()
            if self.verbose:
                print(f"t={self.t} \t state = {self.state} \t action = {self.action}")
            self.take_action()
        if self.t == episode_length_timeout:
            print("Failed to terminate")
        if self.verbose:
            print(f"t={self.t} \t state = {self.state} (terminal)")