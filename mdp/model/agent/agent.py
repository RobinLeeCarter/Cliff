from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import math

if TYPE_CHECKING:
    from mdp.model.environment.state import State
    from mdp.model.environment.action import Action
    from mdp.model.environment.environment import Environment
from mdp import common
# renamed to avoid name conflicts
from mdp.model.algorithm.abstract.algorithm import Algorithm
from mdp.model.algorithm.abstract.episodic import Episodic
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
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

        self._algorithm: Optional[Algorithm] = None
        self._episode: Optional[Episode] = None
        self._record_first_visits: bool = False
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

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
        self._dual_policy_relationship = settings.dual_policy_relationship
        if self._dual_policy_relationship == common.DualPolicyRelationship.SINGLE_POLICY:
            self._policy = primary_policy
            self._behaviour_policy = primary_policy
        elif self._dual_policy_relationship == common.DualPolicyRelationship.INDEPENDENT_POLICIES:
            self._policy = primary_policy
            self._behaviour_policy = policy_factory.policy_factory(self._environment,
                                                                   settings.behaviour_policy_parameters)
        elif self._dual_policy_relationship == common.DualPolicyRelationship.LINKED_POLICIES:
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
        while not self.is_terminal and self.t < episode_length_timeout:
            self.choose_action()
            if self._verbose:
                state: State = self._environment.states[self.s]
                action: Optional[Action]
                if self.a == -1:
                    action = None
                else:
                    action = self._environment.actions[self.a]
                print(f"t={self.t} \t state = {state} \t action = {action}")
            self.take_action()

        if self.t == episode_length_timeout:
            print("Failed to terminate")
        if self._verbose:
            state: State = self._environment.states[self.s]
            print(f"t={self.t} \t state = {state} (terminal)")
        return self._episode

    def start_episode(self, exploring_starts: bool = False):
        """Gets initial state and sets initial reward to None"""
        env = self._environment

        if self._verbose:
            print("start episode...")
        self.t = 0

        self._episode = Episode(env, self.gamma, self._step_callback, self._record_first_visits)

        if exploring_starts:
            # completely random starting state and action and take the action, reward will be None
            # state, action = self._environment.get_random_state_action()
            self.s, self.a = env.s_a_distribution.draw_one()
            self.is_terminal = env.is_terminal[self.s]
            self.r = 0.0
            # action = self._environment.dynamics.get_random_action_for_state(self.state)
            self.choose_action(self.a)
            self.take_action()
        else:
            # get starting state, reward will be None
            self.s = env.start_s_distribution.draw_one()
            # self.s = env.start_s()
            self.is_terminal = env.is_terminal[self.s]
            self.r = 0.0

    def choose_action(self, a: Optional[int] = None):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        if a is None:
            self.a = self._behaviour_policy[self.s]
        else:
            self.a = a
        # is_terminal = self._environment.states[self.s].is_terminal
        self._episode.add_rsa(self.r, self.s, self.a, self.is_terminal)
        if self._verbose:
            state: State = self._environment.states[self.s]
            action: Optional[Action]
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

    # @profile
    def update_target_policy(self, s: int, a: int):
        if self._dual_policy_relationship == common.DualPolicyRelationship.LINKED_POLICIES:
            self._behaviour_policy[s] = a   # this will also update the target policy since linked
        else:
            # in either possible case here we want to update the target policy
            self._policy[s] = a

    def apply_result(self, settings: common.Settings, result: common.Result):
        self.apply_settings(settings)
        self._policy.set_policy_vector(result.policy_vector)
        if self._algorithm.V:
            self._algorithm.V.vector = result.v_vector
        if self._algorithm.Q:
            self._algorithm.Q.set_matrix(result.q_matrix)

    def print_statistics(self):
        self._algorithm.print_q_coverage_statistics()

    def rms_error(self) -> float:
        # better that it just fail if you use something with no V or an environment without get_optimum
        # if not self._algorithm.V or not hasattr(self._environment, 'get_optimum'):
        #     return None

        squared_error: float = 0.0
        count: int = 0
        for s, state in enumerate(self._environment.states):
            if self._environment.is_valued_state(state):
                value: float = self._algorithm.V[s]
                # noinspection PyUnresolvedReferences
                optimum: float = self._environment.get_optimum(state)
                squared_error += (value - optimum)**2
                count += 1
        rms_error = math.sqrt(squared_error / count)
        return rms_error
