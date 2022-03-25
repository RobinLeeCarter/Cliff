from __future__ import annotations
from typing import Optional, TYPE_CHECKING, TypeVar, Generic
from abc import ABC, abstractmethod
import multiprocessing


if TYPE_CHECKING:
    from mdp.controller.base_controller import BaseController
    from mdp.model.base.environment.base_environment import BaseEnvironment
    from mdp.model.base.agent.base_episode import BaseEpisode
    from mdp.model.breakdown.base_breakdown import BaseBreakdown
    from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm
import utils
from mdp import common
from mdp.model.base.environment.base_environment import BaseEnvironment
from mdp.model.base.agent.base_agent import BaseAgent
from mdp.model.breakdown.breakdown_factory import BreakdownFactory
from mdp.model.trainer.trainer import Trainer
from mdp.model.trainer.parallel_trainer import ParallelTrainer


Environment = TypeVar('Environment', bound=BaseEnvironment)
Agent = TypeVar('Agent', bound=BaseAgent)


class BaseModel(Generic[Environment, Agent], ABC):
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self.environment: Optional[Environment] = None
        self.agent: Optional[Agent] = None

        self._controller: Optional[BaseController] = None
        self._comparison: Optional[common.Comparison] = None
        self._breakdown_factory: BreakdownFactory = BreakdownFactory()
        self.breakdown: Optional[BaseBreakdown] = None
        self.trainer: Optional[Trainer] = None
        self.parallel_trainer: Optional[ParallelTrainer] = None

        self._cont: bool = True

    def set_controller(self, controller: BaseController):
        self._controller: BaseController = controller

    def build(self, comparison: common.Comparison):
        self._comparison: common.Comparison = comparison

        # different for each scenario and environment_parameters
        self.environment: Environment = self._create_environment(self._comparison.environment_parameters)
        self.environment.build()
        # self.environment = environment_factory.environment_factory(self._comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent: Agent = self._create_agent()
        # self.agent: Agent = Agent[State, Action](self.environment)

        # breakdowns themselves need comparison in current implementation so breakdown_parameters is not passed in
        if self._comparison.breakdown_parameters:
            self.breakdown = self._breakdown_factory.create(self._comparison)
        self.trainer: Trainer = Trainer(
            agent=self.agent,
            breakdown=self.breakdown,
            model_step_callback=self._display_step,
            verbose=False
        )
        if self.breakdown:
            self.breakdown.set_trainer(self.trainer)
        if self._comparison.settings_list_multiprocessing != common.ParallelContextType.NONE:
            self.parallel_trainer = ParallelTrainer(self.trainer, self._comparison.settings_list_multiprocessing)

    @abstractmethod
    def _create_environment(self, environment_parameters: common.EnvironmentParameters) -> Environment:
        pass

    @abstractmethod
    def _create_agent(self) -> Agent:
        pass

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        if self._comparison.settings_list_multiprocessing == common.ParallelContextType.NONE \
                or multiprocessing.current_process().daemon:
            # train in serial
            for settings in self._comparison.settings_list:
                self.trainer.train(settings)
                if not self._cont:
                    break
        else:
            # train in parallel
            self.parallel_trainer.train(self._comparison.settings_list)
        timer.stop()

        if self.breakdown:
            self.breakdown.compile()

    # def prep_for_output(self):
    #     self.environment.output_mode()
    #     self.switch_to_target_policy()
    #     self.update_grid_value_functions()

    def switch_agent_to_target_policy(self):
        self.agent.set_behaviour_policy(self.algorithm.target_policy)

    def _display_step(self, episode: Optional[BaseEpisode]):
        raise Exception("_display_step() not implemented")
        # self.update_grid_value_functions()
        # self._controller.display_step(episode)

    @property
    def algorithm(self) -> BaseAlgorithm:
        return self.trainer.algorithm
