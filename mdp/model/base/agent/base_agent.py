from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mdp.model.base.environment.base_environment import BaseEnvironment
from mdp import common
from mdp.model.base.agent.base_episode import BaseEpisode
from mdp.model.base.policy.base_policy import BasePolicy


class BaseAgent(ABC):
    def __init__(self,
                 environment: BaseEnvironment,
                 verbose: bool = False):
        self._environment: BaseEnvironment = environment
        self._verbose: bool = verbose

        # self._policy_factory: PolicyFactory = PolicyFactory[State, Action](self._environment)
        self._behaviour_policy: Optional[BasePolicy] = None     # if on-policy = self._policy
        self._dual_policy_relationship: Optional[common.DualPolicyRelationship] = None

        self._episode: Optional[BaseEpisode] = None
        self._first_visit: bool = False
        self._episode_length_timeout: Optional[int] = None

        # not None to avoid unboxing cost of Optional
        self.gamma: float = 1.0
        self.t: int = 0

        # trainer callback
        self._step_callback: Optional[Callable[[], bool]] = None

    @property
    def environment(self) -> BaseEnvironment:
        return self._environment

    @property
    @abstractmethod
    def behaviour_policy(self) -> BasePolicy:
        return self._behaviour_policy

    @property
    @abstractmethod
    def episode(self) -> BaseEpisode:
        return self._episode

    def apply_settings(self, settings: common.Settings):
        # top-down
        self._episode_length_timeout: int = settings.episode_length_timeout
        self.gamma: float = settings.gamma

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
    def _print_step(self):
        ...
