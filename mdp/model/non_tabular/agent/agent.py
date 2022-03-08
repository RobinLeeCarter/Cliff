from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable, TypeVar

# import math

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

from mdp import common
# renamed to avoid name conflicts
from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm
from mdp.model.tabular.algorithm.abstract.episodic import Episodic
from mdp.model.non_tabular.agent.episode import Episode
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy

from mdp.model.general.algorithm import algorithm_factory
from mdp.model.general.policy import policy_factory

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class Agent:
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 verbose: bool = False):
        self._environment: NonTabularEnvironment[State, Action] = environment
        self._verbose: bool = verbose

        self._policy: Optional[NonTabularPolicy[State, Action]] = None
        self._behaviour_policy: Optional[NonTabularPolicy[State, Action]] = None     # if on-policy = self._policy
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

        self._algorithm: Optional[Algorithm] = None
        self._episode: Optional[Episode] = None
        self._record_first_visits: bool = False
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

        # always refers to values for time-step t
        self.state: Optional[State] = None
        self.action: Optional[Action] = None

        self.r: float = 0.0
        # self.is_terminal: bool = False  # stored for performance

        # always refers to values for time-step t-1
        self.prev_r: float = 0.0
        self.prev_state: Optional[State] = None
        self.prev_action: Optional[Action] = None

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    # use for on-policy algorithms
    @property
    def policy(self) -> NonTabularPolicy[State, Action]:
        return self._policy

    # use these two for off-policy algorithms
    @property
    def target_policy(self) -> NonTabularPolicy[State, Action]:
        return self._policy

    @property
    def behaviour_policy(self) -> NonTabularPolicy[State, Action]:
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
            environment=self._environment,
            agent=self,
            algorithm_parameters=settings.algorithm_parameters)
        settings.algorithm_title = self._algorithm.title
        self._episode_length_timeout = settings.episode_length_timeout
        if isinstance(self._algorithm, Episodic):
            self._record_first_visits = self._algorithm.first_visit
        else:
            self._record_first_visits = False

        self.gamma = settings.gamma

    def set_behaviour_policy(self, policy: NonTabularPolicy[State, Action]):
        self._behaviour_policy = policy

    def parameter_changes(self, iteration: int):
        # potentially change epsilon here
        self._algorithm.parameter_changes(iteration)

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    def generate_episodes(self,
                          num_episodes: int,
                          episode_length_timeout: Optional[int] = None
                          ) -> list[Episode]:
        return [self.generate_episode(episode_length_timeout)
                for _ in range(num_episodes)]

    def generate_episode(self,
                         episode_length_timeout: Optional[int] = None,
                         ) -> Episode:
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
        env = self._environment

        if self._verbose:
            print("start episode...")
        self.t = 0

        self._episode = Episode(env, self.gamma, self._step_callback, self._record_first_visits)

        # get starting state, reward will be None
        self.state = self._environment.draw_start_state()
        self.action = None
        # self.s = env.start_s()
        # self.is_terminal = self.state.is_terminal
        self.r = 0.0

    def choose_action(self, forced_action: Optional[Action] = None):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """
        if forced_action:
            self.action = forced_action
        else:
            self.action = self._behaviour_policy[self.state]

        self._episode.add_rsa(self.r, self.state, self.action)
        if self._verbose:
            print(f"state = {self.state} \t action = {self.action}")

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
            self._episode.add_rsa(self.r, self.state, self.action)

    def _print_step(self):
        print(f"t={self.t} \t state = {self.state} \t action = {self.action}")

    def print_statistics(self):
        self._algorithm.print_q_coverage_statistics()

    def rms_error(self) -> float:
        raise NotImplementedError
    #     # better that it just fail if you use something with no V or an environment without get_optimum
    #     # if not self._algorithm.V or not hasattr(self._environment, 'get_optimum'):
    #     #     return None
    #
    #     squared_error: float = 0.0
    #     count: int = 0
    #     for s, state in enumerate(self._environment.states):
    #         if self._environment.is_valued_state(state):
    #             value: float = self._algorithm.V[s]
    #             # noinspection PyUnresolvedReferences
    #             optimum: float = self._environment.get_optimum(state)
    #             squared_error += (value - optimum)**2
    #             count += 1
    #     rms_error = math.sqrt(squared_error / count)
    #     return rms_error
