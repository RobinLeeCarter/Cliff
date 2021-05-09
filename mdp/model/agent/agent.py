from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import math

if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.response import Response
    from mdp.model.environment.environment import Environment
from mdp import common
# renamed to avoid name conflicts
from mdp.model.algorithm.abstract.algorithm_ import Algorithm
from mdp.model.algorithm.abstract.episodic_ import Episodic
from mdp.model.agent.episode import Episode
from mdp.model.policy.policy import Policy
from mdp.model.algorithm import algorithm_factory
from mdp.model.policy import policy_factory


class Agent:
    def __init__(self,
                 environment_: Environment,
                 verbose: bool = False):
        self._environment: Environment = environment_
        self._verbose: bool = verbose

        self._policy: Optional[Policy] = None
        self._behaviour_policy: Optional[Policy] = None     # if on-policy = self._policy
        self._algorithm: Optional[Algorithm] = None
        self._episode: Optional[Episode] = None
        self._record_first_visits: bool = False
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

        # always refers to values for time-step _t
        self.reward: Optional[float] = None
        self.state: Optional[State] = None
        self.action: Optional[Action] = None

        # always refers to values for time-step _t-1
        self.prev_reward: Optional[float] = None
        self.prev_state: Optional[State] = None
        self.prev_action: Optional[Action] = None

        self._response: Optional[Response] = None

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    # use for on-policy algorithms
    @property
    def policy(self) -> Policy:
        return self._policy

    # use these two for off-policy algorithms
    @property
    def target_policy(self) -> Policy:
        return self._policy

    @property
    def behaviour_policy(self) -> Policy:
        return self._behaviour_policy

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @property
    def episode(self) -> Episode:
        return self._episode

    def apply_settings(self, settings: common.Settings):
        # sort out policies
        primary_policy = policy_factory.policy_factory(self._environment, settings.policy_parameters)
        r: common.DualPolicyRelationship = settings.dual_policy_relationship
        if r == common.DualPolicyRelationship.SINGLE_POLICY:
            self._policy = primary_policy
            self._behaviour_policy = primary_policy
        elif r == common.DualPolicyRelationship.INDEPENDENT_POLICIES:
            self._policy = primary_policy
            self._behaviour_policy = policy_factory.policy_factory(self._environment,
                                                                   settings.behaviour_policy_parameters)
        elif r == common.DualPolicyRelationship.LINKED_POLICIES:
            self._policy = primary_policy.linked_policy     # typically the deterministic part we want to output
            self._behaviour_policy = primary_policy
        else:
            raise NotImplementedError

        # set policy based on policy_parameters
        self._algorithm = algorithm_factory.algorithm_factory(
            environment_=self._environment,
            agent_=self,
            algorithm_parameters=settings.algorithm_parameters,
            policy_parameters=settings.policy_parameters)
        self._episode_length_timeout = settings.episode_length_timeout
        if isinstance(self._algorithm, Episodic):
            self._record_first_visits = self._algorithm.first_visit
        else:
            self._record_first_visits = False

        self.gamma = settings.gamma

    def set_behaviour_policy(self, policy: Policy):
        self._behaviour_policy = policy

    # def initialize(self):
    #     # initialize policies here? pass in settings too?
    #     self._algorithm.initialize()

    def parameter_changes(self, iteration: int):
        # potentially change epsilon here
        self._algorithm.parameter_changes(iteration)

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    def generate_episode(self,
                         episode_length_timeout: Optional[int] = None,
                         exploring_starts: bool = False
                         ) -> Episode:
        if not episode_length_timeout:
            episode_length_timeout = self._episode_length_timeout

        self.start_episode(exploring_starts)
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

    def start_episode(self, exploring_starts: bool = False):
        """Gets initial state and sets initial reward to None"""
        if self._verbose:
            print("start episode...")
        self.t = 0

        self._episode = Episode(self.gamma, self._step_callback, self._record_first_visits)

        if exploring_starts:
            # completely random starting state and action, reward will be None
            state, action = self._environment.dynamics.get_random_state_action()
            self.state = state
            self.reward = None
            # action = self._environment.dynamics.get_random_action_for_state(self.state)
            self.choose_action(action)
            self.take_action()
        else:
            # get starting state, reward will be None
            self._response = self._environment.start()
            self.reward = self._response.reward
            self.state = self._response.state

    def choose_action(self, action: Optional[Action] = None):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        if action:
            self.action = action
        else:
            self.action = self._behaviour_policy[self.state]
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

    def print_statistics(self):
        self._algorithm.print_q_coverage_statistics()

    def rms_error(self) -> float:
        # better that it just fail if you use something with no V or an environment without get_optimum
        # if not self._algorithm.V or not hasattr(self._environment, 'get_optimum'):
        #     return None

        squared_error: float = 0.0
        count: int = 0
        for state in self._environment.states:
            if self._environment.is_valued_state(state):
                value: float = self._algorithm.V[state]
                # noinspection PyUnresolvedReferences
                optimum: float = self._environment.get_optimum(state)
                squared_error += (value - optimum)**2
                count += 1
        rms_error = math.sqrt(squared_error / count)
        return rms_error
