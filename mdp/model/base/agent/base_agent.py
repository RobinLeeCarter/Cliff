from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mdp.model.base.environment.base_environment import BaseEnvironment
from mdp import common
from mdp.model.base.algorithm.algorithm_factory import AlgorithmFactory
from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm
from mdp.model.base.agent.base_episode import BaseEpisode
from mdp.model.base.policy.base_policy import BasePolicy


class BaseAgent(ABC):
    def __init__(self,
                 environment: BaseEnvironment,
                 verbose: bool = False):
        self._environment: BaseEnvironment = environment
        self._verbose: bool = verbose

        # self._policy_factory: PolicyFactory = PolicyFactory[State, Action](self._environment)
        self._policy: Optional[BasePolicy] = None
        self._behaviour_policy: Optional[BasePolicy] = None     # if on-policy = self._policy
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

        self._algorithm_factory: Optional[AlgorithmFactory] = None
        self._algorithm: Optional[BaseAlgorithm] = None
        self._episode: Optional[BaseEpisode] = None
        self._record_first_visits: bool = False
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    @property
    def environment(self) -> BaseEnvironment:
        return self._environment

    # use these two for off-policy algorithms
    @property
    @abstractmethod
    def target_policy(self) -> BasePolicy:
        return self._policy

    @property
    @abstractmethod
    def behaviour_policy(self) -> BasePolicy:
        return self._behaviour_policy

    @property
    def algorithm_factory(self) -> AlgorithmFactory:
        return self._algorithm_factory

    @property
    @abstractmethod
    def algorithm(self) -> BaseAlgorithm:
        return self._algorithm

    @property
    @abstractmethod
    def episode(self) -> BaseEpisode:
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

    def set_behaviour_policy(self, policy: BasePolicy):
        self._behaviour_policy = policy

    def set_step_callback(self, step_callback: Optional[Callable[[], bool]] = None):
        self._step_callback = step_callback

    def generate_episodes(self,
                          num_episodes: int,
                          episode_length_timeout: Optional[int] = None
                          ) -> list[BaseEpisode]:
        return [self.generate_episode(episode_length_timeout)
                for _ in range(num_episodes)]

    @abstractmethod
    def generate_episode(self,
                         episode_length_timeout: Optional[int] = None,
                         ) -> BaseEpisode:
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
