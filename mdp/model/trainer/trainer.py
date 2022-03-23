from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import multiprocessing

if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
    from mdp.model.base.agent.base_episode import BaseEpisode
    from mdp.model.breakdown.base_breakdown import BaseBreakdown
from mdp import common
from mdp.model.base.policy.policy_factory import PolicyFactory
from mdp.model.base.algorithm.algorithm_factory import AlgorithmFactory
from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm
from mdp.model.tabular.algorithm.abstract.episodic import Episodic
from mdp.model.tabular.algorithm.abstract.dynamic_programming import DynamicProgramming
from mdp.model.trainer.parallel_runner import ParallelRunner


class Trainer:
    def __init__(self,
                 agent: BaseAgent,
                 breakdown: Optional[BaseBreakdown],
                 model_step_callback: Optional[Callable[[Optional[BaseEpisode]], None]] = None,
                 verbose: bool = False
                 ):
        self._agent: BaseAgent = agent
        self._breakdown: Optional[BaseBreakdown] = breakdown
        self.settings: Optional[common.Settings] = None

        self._model_step_callback: Optional[Callable[[Optional[BaseEpisode]], None]] = model_step_callback
        self._verbose = verbose
        self._cont: bool = True

        self._algorithm_factory: Optional[AlgorithmFactory] = AlgorithmFactory(self._agent)
        self._algorithm: Optional[BaseAlgorithm] = None
        self._policy_factory: PolicyFactory = PolicyFactory(self._agent.environment)

        self.run_counter: int = 0
        self.episode_counter: int = 0

        self.cum_timestep: int = 0  # cumulative timestep across all episodes for a given run
        self.max_cum_timestep: int = 0  # max cumulative timestep across all runs

    @property
    def breakdown(self) -> BaseBreakdown:
        return self._breakdown

    @property
    def algorithm_factory(self) -> AlgorithmFactory:
        return self._algorithm_factory

    @property
    def algorithm(self) -> BaseAlgorithm:
        return self._algorithm

    @property
    def agent(self) -> BaseAgent:
        return self._agent

    @property
    def episode(self) -> BaseEpisode:
        return self._agent.episode

    def disable_step_callback(self):
        self._model_step_callback = None

    def train(self, settings: common.Settings, return_result: bool = False) -> Optional[common.Result]:
        self.apply_settings(settings)

        match self._algorithm:
            case Episodic():
                self._train_episodic()
            case DynamicProgramming():
                self._train_dynamic_programming()
            case _:
                raise NotImplementedError

        if settings.algorithm_parameters.derive_v_from_q_as_final_step:
            assert isinstance(self._algorithm, TabularAlgorithm)
            self._algorithm.derive_v_from_q()

        if return_result:
            return self._get_result(settings.result_parameters)

    def apply_settings(self, settings: common.Settings):
        self.settings = settings
        self._algorithm = self._algorithm_factory.create(settings.algorithm_parameters)
        self._algorithm.create_policies(self._policy_factory, settings)
        self._agent.apply_settings(self.settings)

    def _train_episodic(self):
        settings = self.settings
        if (settings.review_every_step or settings.display_every_step) and self._agent.set_step_callback:
            self._agent.set_step_callback(self.step)
        title: str = self._algorithm_factory.get_algorithm_title(settings.algorithm_parameters)
        print(f"{title}: {settings.runs} runs")

        self.max_cum_timestep = 0
        if settings.runs_multiprocessing == common.ParallelContextType.NONE \
                or multiprocessing.current_process().daemon:
            # train in serial if not multiprocessing or if a child process (daemon)
            for run_counter in range(1, settings.runs + 1):
                self.do_run(run_counter)
                self.max_cum_timestep = max(self.max_cum_timestep, self.cum_timestep)
        else:
            # train in parallel
            self.parallel_runner = ParallelRunner(self)
            self.parallel_runner.do_runs()

        if self._verbose:
            if isinstance(self._algorithm, TabularAlgorithm):
                self._algorithm.print_q_coverage_statistics()

    def do_run(self,
               run_counter: int,
               result_parameters: Optional[common.ResultParameters] = None
               ) -> Optional[common.Result]:
        """Perform a single run. Could be passed with overridden settings but agent will be unchanged"""
        # for use by Breakdown
        self.run_counter = run_counter

        settings = self.settings
        if self._verbose or run_counter % settings.run_print_frequency == 0:
            print(f"run_counter = {run_counter}: {settings.training_episodes} episodes")

        self._algorithm.initialize()

        self.cum_timestep = 0
        # TODO: optionally generate and pre-process episodes in parallel e.g. for Blackjack
        #  according to a behavioural policy which then might change so perhaps in batches
        for episode_counter in range(1, settings.training_episodes + 1):
            self._do_episode(episode_counter)

        if result_parameters:
            return self._get_result(result_parameters)

    def _do_episode(self, episode_counter: int):
        assert isinstance(self._algorithm, Episodic)

        # for use by Breakdown
        self.episode_counter = episode_counter
        if self._verbose or episode_counter % self.settings.episode_print_frequency == 0:
            print(f"episode_counter = {episode_counter}")

        if not self.settings.review_every_step and self.cum_timestep != 0:
            self.cum_timestep += 1  # start next episode from the next timestep

        self._algorithm.parameter_changes(episode_counter)
        self._algorithm.do_episode(self.settings.episode_length_timeout)
        episode = self._agent.episode

        if self._verbose:
            max_t = episode.max_t
            total_return = episode.total_return
            if self._verbose:
                print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
        if not self.settings.review_every_step:
            self.cum_timestep += episode.max_t
            if self._breakdown:
                self._breakdown.review()

    def _train_dynamic_programming(self):
        assert isinstance(self._algorithm, DynamicProgramming)
        if self.settings.review_every_step or self.settings.display_every_step:
            self._algorithm.set_step_callback(self.step)
        self._algorithm.initialize()
        self._algorithm.run()

    def step(self) -> bool:
        if self.settings.review_every_step:
            self._review_step()
        if self.settings.display_every_step and self._model_step_callback:
            self._model_step_callback(self._agent.episode)
        return True

    def _review_step(self):
        self.cum_timestep += 1
        self._breakdown.review()

    def _get_result(self, result_parameters: common.ResultParameters):
        """Build up Result object, deciding on what to include by referring to result_parameters"""
        result: common.Result = common.Result()

        rp: common.ResultParameters = result_parameters

        if rp.return_recorder and self._breakdown:
            result.recorder = self._breakdown.recorder

        if rp.return_algorithm_title:
            result.algorithm_title = self._algorithm.title

        if isinstance(self._algorithm, TabularAlgorithm):
            if rp.return_policy_vector:
                result.policy_vector = self._algorithm.target_policy.get_policy_vector()
            if rp.return_v_vector and self._algorithm.V:
                result.v_vector = self._algorithm.V.vector
            if rp.return_q_matrix and self._algorithm.Q:
                result.q_matrix = self._algorithm.Q.matrix

        if rp.return_cum_timestep:
            result.cum_timestep = self.cum_timestep

        return result
