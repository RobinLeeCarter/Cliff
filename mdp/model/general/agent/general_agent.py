from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Callable    # , TypeVar, Generic

# import math

if TYPE_CHECKING:
    from mdp.model.general.environment.general_environment import GeneralEnvironment
    # from mdp.model.general.environment.general_state import GeneralState
    from mdp.model.general.environment.general_action import GeneralAction

from mdp import common
# renamed to avoid name conflicts
from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm
# from mdp.model.tabular.algorithm.abstract.episodic import Episodic
from mdp.model.general.agent.general_episode import GeneralEpisode
# from mdp.model.non_tabular.policy.policy_factory import PolicyFactory
from mdp.model.general.policy.general_policy import GeneralPolicy

from mdp.model.general.algorithm import algorithm_factory
# from mdp.model.general.policy import policy_factory

# State = TypeVar('State', bound=GeneralState)
# Action = TypeVar('Action', bound=GeneralAction)
# Policy = TypeVar('Policy', bound=GeneralPolicy)


class GeneralAgent(ABC):
    def __init__(self,
                 environment: GeneralEnvironment,
                 verbose: bool = False):
        self._environment: GeneralEnvironment = environment
        self._verbose: bool = verbose

        # self._policy_factory: PolicyFactory = PolicyFactory[State, Action](self._environment)
        self._policy: Optional[GeneralPolicy] = None
        self._behaviour_policy: Optional[GeneralPolicy] = None     # if on-policy = self._policy
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

        self._algorithm: Optional[Algorithm] = None
        self._episode: Optional[GeneralEpisode] = None
        self._record_first_visits: bool = False
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    # use for on-policy algorithms
    @property
    @abstractmethod
    def policy(self) -> GeneralPolicy:
        return self._policy

    # use these two for off-policy algorithms
    @property
    @abstractmethod
    def target_policy(self) -> GeneralPolicy:
        return self._policy

    @property
    @abstractmethod
    def behaviour_policy(self) -> GeneralPolicy:
        return self._behaviour_policy

    @property
    @abstractmethod
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @property
    @abstractmethod
    def episode(self) -> GeneralEpisode:
        return self._episode

    @abstractmethod
    def apply_settings(self, settings: common.Settings):
        pass
        # sort out policies
        # primary_policy = self._policy_factory.create(settings.policy_parameters)
        # self._dual_policy_relationship = settings.dual_policy_relationship
        # if self._dual_policy_relationship == common.DualPolicyRelationship.SINGLE_POLICY:
        #     self._policy = primary_policy
        #     self._behaviour_policy = primary_policy
        # elif self._dual_policy_relationship == common.DualPolicyRelationship.INDEPENDENT_POLICIES:
        #     self._policy = primary_policy
        #     self._behaviour_policy = self._policy_factory.create(settings.behaviour_policy_parameters)
        # elif self._dual_policy_relationship == common.DualPolicyRelationship.LINKED_POLICIES:
        #     self._policy = primary_policy.linked_policy     # typically the deterministic part we want to output
        #     self._behaviour_policy = primary_policy
        # else:
        #     raise NotImplementedError
        #
        # # set policy based on policy_parameters
        # self._algorithm = algorithm_factory.algorithm_factory(
        #     environment=self._environment,
        #     agent=self,
        #     algorithm_parameters=settings.algorithm_parameters)
        # settings.algorithm_title = self._algorithm.title
        # self._episode_length_timeout = settings.episode_length_timeout
        # if isinstance(self._algorithm, Episodic):
        #     self._record_first_visits = self._algorithm.first_visit
        # else:
        #     self._record_first_visits = False
        #
        # self.gamma = settings.gamma

    def set_behaviour_policy(self, policy: GeneralPolicy):
        self._behaviour_policy = policy

    def parameter_changes(self, iteration: int):
        # potentially change epsilon here
        self._algorithm.parameter_changes(iteration)

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    def generate_episodes(self,
                          num_episodes: int,
                          episode_length_timeout: Optional[int] = None
                          ) -> list[GeneralEpisode]:
        return [self.generate_episode(episode_length_timeout)
                for _ in range(num_episodes)]

    @abstractmethod
    def generate_episode(self,
                         episode_length_timeout: Optional[int] = None,
                         ) -> GeneralEpisode:
        ...

    @abstractmethod
    def start_episode(self):
        """Gets initial state and sets initial reward to None"""

    @abstractmethod
    def choose_action(self):
        """
        Have the policy choose an action
        We then have a complete r, s, a to add to episode
        The reward being is response from the previous action (if there was one, or otherwise reward=None)
        Note that the action is NOT applied yet.
        """

    @abstractmethod
    def take_action(self):
        """With state and action are already set,
        Perform action.
        Get new reward and state in response.
        Start a new time step with the new reward and state
        """

    @abstractmethod
    def apply_result(self, result: common.Result):
        ...

    @abstractmethod
    def _print_step(self):
        ...

    @abstractmethod
    def print_statistics(self):
        ...

    @abstractmethod
    def rms_error(self) -> float:
        ...
