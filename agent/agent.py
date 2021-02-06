from typing import Optional

import constants
import environment
import policy
from agent import episode, rsa, sarsa


class Agent:
    def __init__(self, environment_: environment.Environment, policy_: policy.Policy, verbose: bool = False):
        self.environment: environment.Environment = environment_
        self.policy: policy.Policy = policy_
        self.verbose: bool = verbose
        self.episode: Optional[episode.Episode] = None

        self.previous_rsa: Optional[rsa.RSA] = None
        self.state: Optional[environment.State] = None
        self.action: Optional[environment.Action] = None
        self.reward: Optional[float] = None
        self.response: Optional[environment.Observation] = None
        self.t: Optional[int] = None

    def set_policy(self, policy_: policy.Policy):
        self.policy = policy_

    def start_episode(self):
        """Gets initial state S0.
        Choose initial action A0.
        Records S0 and A0 but does not take action."""
        if self.verbose:
            print("start episode...")
        self.episode = episode.Episode()
        self.t = 0
        # self.reward = None
        self.previous_rsa = None
        # start
        self.response = self.environment.start()
        self.reward = self.response.reward
        self.state = self.response.state
        # special case at start
        # self._add_episode_rsa()

    def take_action(self):
        """State and action are already set, make a copy in previous_rsa before updating.
        Perform action.
        Get new reward and state in response.
        Update current values including new action.
        Record new reward, state and action in episode.
        """
        if self.state.is_terminal:
            raise Exception("Trying to act in a terminal state.")
        self.previous_rsa = self.episode.rsa
        self.response = self.environment.from_state_perform_action(self.state, self.action)
        self.t += 1
        self.reward = self.response.reward
        self.state = self.response.state

    def choose_action(self):
        self.action = self.policy[self.state]
        self.episode.add_rsa(reward=self.reward, state=self.state, action=self.action)
        if self.verbose:
            print(f"state = {self.state} \t action = {self.action}")

    def get_sarsa(self) -> sarsa.SARSA:
        if self.previous_rsa is None:
            raise Exception("Trying to get_sarsa with no previous_rsa.")
        sarsa_ = sarsa.SARSA(
            state=self.previous_rsa.state,
            action=self.previous_rsa.action,
            next_reward=self.reward,
            next_state=self.state,
            next_action=self.action
        )
        return sarsa_

    def generate_episode(self):
        self.start_episode()
        while not self.state.is_terminal and self.t < constants.EPISODE_LENGTH_TIMEOUT:
            self.choose_action()
            if self.verbose:
                print(f"t={self.t} \t state = {self.state} \t action = {self.action}")
            self.take_action()
        if self.t == constants.EPISODE_LENGTH_TIMEOUT:
            print("Failed to terminate")
        if self.verbose:
            print(f"t={self.t} \t state = {self.state} (terminal)")
