from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import multiprocessing

from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic_batch import NonTabularEpisodicBatch
from mdp.model.non_tabular.algorithm.episodic.episodic_sarsa_parallel_w import EpisodicSarsaParallelW
from mdp.model.trainer.parallel_episodes import ParallelEpisodes
from mdp.model.trainer.parallel_episodes_w import ParallelEpisodesW

if TYPE_CHECKING:
    from mdp.model.base.agent.base_agent import BaseAgent
    from mdp.model.base.environment.base_environment import BaseEnvironment
    from mdp.model.base.agent.base_episode import BaseEpisode
    from mdp.model.breakdown.base_breakdown import BaseBreakdown
from mdp import common
from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment

from mdp.factory.algorithm_factory import AlgorithmFactory
from mdp.factory.policy_factory import PolicyFactory
from mdp.factory.feature_factory import FeatureFactory
from mdp.factory.value_function_factory import ValueFunctionFactory

from mdp.model.base.algorithm.base_algorithm import BaseAlgorithm
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm
from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm
from mdp.model.tabular.algorithm.abstract.tabular_episodic import TabularEpisodic
from mdp.model.non_tabular.algorithm.abstract.nontabular_episodic import NonTabularEpisodic
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
        environment: BaseEnvironment = self._agent.environment
        self._breakdown: Optional[BaseBreakdown] = breakdown
        self.settings: Optional[common.Settings] = None

        self._model_step_callback: Optional[Callable[[Optional[BaseEpisode]], None]] = model_step_callback
        self._verbose = verbose
        self._cont: bool = True

        self._algorithm_factory: Optional[AlgorithmFactory] = AlgorithmFactory(self._agent)
        self._algorithm: Optional[BaseAlgorithm] = None
        self._policy_factory: PolicyFactory = PolicyFactory(environment)

        if isinstance(environment, NonTabularEnvironment):
            self._feature_factory: FeatureFactory = FeatureFactory(environment.dims)
            self._value_function_factory: ValueFunctionFactory = ValueFunctionFactory()

        self._parallel_runner: Optional[ParallelRunner] = None
        self._parallel_episodes: Optional[ParallelEpisodes] = None

        self.run_counter: int = 0
        self.episode_counter: int = 0

        self.cum_timestep: int = 0  # cumulative timestep across all episodes for a given run
        self.max_cum_timestep: int = 0  # max cumulative timestep across all runs

    @property
    def breakdown(self) -> Optional[BaseBreakdown]:
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

        daemon: bool = multiprocessing.current_process().daemon
        if settings.runs_multiprocessing and not daemon:
            # do runs in parallel
            self._parallel_runner = ParallelRunner(self)
        else:
            # do runs in serial
            self._parallel_runner = None
            if settings.episode_multiprocessing \
                    and self._algorithm.batch_episodes \
                    and not daemon:
                # do episodes in parallel
                if isinstance(self.algorithm, EpisodicSarsaParallelW):
                    self._parallel_episodes = ParallelEpisodesW(self)
                else:
                    self._parallel_episodes = ParallelEpisodes(self)
            else:
                # do episodes in serial (with batch determined by self._algorithm.batch_episodes)
                self._parallel_episodes = None

        if self._algorithm.episodic:
            self._train_episodic()
        elif self._algorithm.dynamic_programming:
            self._train_dynamic_programming()
        else:
            raise NotImplementedError

        if self._algorithm.tabular and settings.algorithm_parameters.derive_v_from_q_as_final_step:
            assert isinstance(self._algorithm, TabularAlgorithm)
            self._algorithm.derive_v_from_q()

        if return_result:
            return self._get_result(settings.result_parameters)

    def apply_settings(self, settings: common.Settings):
        self.settings = settings
        self._algorithm = self._algorithm_factory.create(settings.algorithm_parameters)
        self._algorithm.apply_settings(settings)
        self._algorithm.create_policies(self._policy_factory, settings)
        if not self._algorithm.tabular:
            assert isinstance(self._algorithm, NonTabularAlgorithm)
            self._algorithm.create_feature_and_value_function(
                self._feature_factory, self._value_function_factory, settings)

    def _train_episodic(self):
        settings = self.settings
        if (settings.review_every_step or settings.display_every_step) and self._agent.set_step_callback:
            self._agent.set_step_callback(self.step)
        # title: str = self._algorithm_factory.get_algorithm_title(settings.algorithm_parameters)
        print(f"{self._algorithm.title}: {settings.runs} runs")

        self.max_cum_timestep = 0
        if self._parallel_runner:
            # train in parallel
            self._parallel_runner.do_runs()
        else:
            # train in serial
            for run_counter in range(1, settings.runs + 1):
                self.do_run(run_counter)
                self.max_cum_timestep = max(self.max_cum_timestep, self.cum_timestep)

        if self._verbose:
            if self._algorithm.tabular:
                assert isinstance(self._algorithm, TabularAlgorithm)
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

        if self._parallel_episodes:
            # train in parallel batches
            actual_episodes_per_batch: int = self._parallel_episodes.actual_episodes_per_batch
            for first_batch_episode in range(1, settings.training_episodes + 1, actual_episodes_per_batch):
                self._parallel_episodes.do_episode_batch(first_batch_episode)
        else:
            # train in serial
            if self._algorithm.batch_episodes:
                # train in batches of episodes
                episodes_per_batch = self.settings.episodes_per_batch
                training_episodes = settings.training_episodes
                for first_batch_episode in range(1, training_episodes + 1, episodes_per_batch):
                    episodes_to_do = min(training_episodes + 1 - first_batch_episode, episodes_per_batch)
                    self.do_episodes(episode_counter_start=first_batch_episode,
                                     episodes_to_do=episodes_to_do)
            else:
                # train one episode at a time
                for episode_counter in range(1, settings.training_episodes + 1):
                    self._do_episode(episode_counter)

        if result_parameters:
            return self._get_result(result_parameters)

    # called from ParallelEpisodes and above
    def do_episodes(self,
                    episode_counter_start: int,
                    episodes_to_do: int,
                    result_parameters: Optional[common.ResultParameters] = None
                    ) -> Optional[common.Result]:
        assert isinstance(self._algorithm, NonTabularEpisodicBatch)
        self._algorithm.start_episodes()
        for episode_counter in range(episode_counter_start, episode_counter_start + episodes_to_do):
            self._do_episode(episode_counter)

        if self._parallel_episodes:
            if result_parameters:
                return self._get_result(result_parameters)
        else:
            self._algorithm.apply_trajectories()

    def _do_episode(self, episode_counter: int):
        # for use by Breakdown
        self.episode_counter = episode_counter
        if self._verbose or episode_counter % self.settings.episode_print_frequency == 0:
            print(f"episode_counter = {episode_counter}")
        if not self.settings.review_every_step and self.cum_timestep != 0:
            self.cum_timestep += 1  # start next episode from the next timestep
        self._algorithm.parameter_changes(episode_counter)

        assert isinstance(self._algorithm, TabularEpisodic) or isinstance(self._algorithm, NonTabularEpisodic)
        episode: BaseEpisode = self._algorithm.do_episode()

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
        if self._breakdown:
            self._breakdown.review()

    def _get_result(self, result_parameters: common.ResultParameters):
        """Build up Result object, deciding on what to include by referring to result_parameters"""
        result: common.Result = common.Result()

        rp: common.ResultParameters = result_parameters

        if rp.return_recorder and self._breakdown:
            result.recorder = self._breakdown.recorder

        if self._algorithm.tabular:
            assert isinstance(self._algorithm, TabularAlgorithm)
            if rp.return_policy_vector:
                result.policy_vector = self._algorithm.target_policy.get_policy_vector()
            if rp.return_v_vector and self._algorithm.V:
                result.v_vector = self._algorithm.V.vector
            if rp.return_q_matrix and self._algorithm.Q:
                result.q_matrix = self._algorithm.Q.matrix

        if rp.return_cum_timestep:
            result.cum_timestep = self.cum_timestep

        if rp.return_delta_w_vector and self._algorithm.batch_episodes:
            assert isinstance(self._algorithm, NonTabularEpisodicBatch)
            result.delta_w_vector = self._algorithm.get_delta_weights()

        if rp.return_trajectories and self._algorithm.batch_episodes:
            assert isinstance(self._algorithm, NonTabularEpisodicBatch)
            result.trajectories = self._algorithm.trajectories

        return result
