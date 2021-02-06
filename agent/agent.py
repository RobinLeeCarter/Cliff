from typing import Optional

import constants
import environment
import policy
from agent import episode, sarsa


class Agent:
    def __init__(self, environment_: environment.Environment, policy_: policy.Policy, verbose: bool = False):
        self.environment: environment.Environment = environment_
        self.policy: policy.Policy = policy_
        self.verbose: bool = verbose
        self.episode: Optional[episode.Episode] = None

        self.reward: Optional[float] = None
        self.state: Optional[environment.State] = None
        self.action: Optional[environment.Action] = None

        self.prev_reward: Optional[float] = None
        self.prev_state: Optional[environment.State] = None
        self.prev_action: Optional[environment.Action] = None

        self.response: Optional[environment.Observation] = None
        self.t: Optional[int] = None

    def set_policy(self, policy_: policy.Policy):
        self.policy = policy_

    def start_episode(self):
        """Gets initial state and sets initial reward to None"""
        if self.verbose:
            print("start episode...")
        self.episode = episode.Episode()
        self.t = 0
        # start
        self.response = self.environment.start()
        self.reward = self.response.reward
        self.state = self.response.state

    def choose_action(self):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is the response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        self.prev_action = self.action
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
        self.prev_reward = self.reward
        self.prev_state = self.state

        self.response = self.environment.from_state_perform_action(self.state, self.action)
        # begin building up the next r, s, a
        self.t += 1
        self.reward = self.response.reward
        self.state = self.response.state

    def get_sarsa(self) -> sarsa.SARSA:
        sarsa_ = sarsa.SARSA(
            prev_state=self.prev_state,
            prev_action=self.prev_action,
            reward=self.reward,
            state=self.state,
            action=self.action
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
