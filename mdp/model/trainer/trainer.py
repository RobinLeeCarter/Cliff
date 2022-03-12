from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import multiprocessing

if TYPE_CHECKING:
    from mdp.model.general.agent.general_agent import GeneralAgent
    from mdp.model.general.agent.general_episode import GeneralEpisode
    from mdp.model.breakdown.breakdown import Breakdown
from mdp import common
from mdp.model.general.algorithm.general_algorithm import GeneralAlgorithm
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm
from mdp.model.tabular.algorithm.abstract.episodic import Episodic
from mdp.model.tabular.algorithm.abstract.dynamic_programming import DynamicProgramming
from mdp.model.tabular.agent.tabular_agent import TabularAgent

from mdp.model.trainer.parallel_runner import ParallelRunner


class Trainer:
    def __init__(self,
                 agent: GeneralAgent,
                 breakdown: Optional[Breakdown],
                 model_step_callback: Optional[Callable[[Optional[GeneralEpisode]], None]] = None,
                 verbose: bool = False
                 ):
        self._agent: GeneralAgent = agent
        self._breakdown: Optional[Breakdown] = breakdown
        self.settings: Optional[common.Settings] = None

        self._model_step_callback: Optional[Callable[[Optional[GeneralEpisode]], None]] = model_step_callback
        self._verbose = verbose
        self._cont: bool = True

        self.run_counter: int = 0
        self.episode_counter: int = 0

        self.cum_timestep: int = 0  # cumulative timestep across all episodes for a given run
        self.max_cum_timestep: int = 0  # max cumulative timestep across all runs

    @property
    def breakdown(self) -> Breakdown:
        return self._breakdown

    @property
    def episode(self) -> GeneralEpisode:
        return self._agent.episode

    @property
    def agent(self) -> GeneralAgent:
        return self._agent

    def disable_step_callback(self):
        self._model_step_callback = None

    def train(self, settings: common.Settings, return_result: bool = False) -> Optional[common.Result]:
        # process settings
        self.settings = settings
        self._agent.apply_settings(self.settings)

        algorithm: GeneralAlgorithm = self._agent.algorithm
        match algorithm:
            case Episodic():
                self._train_episodic()
            case DynamicProgramming():
                self._train_dynamic_programming()
            case _:
                raise NotImplementedError

        # if isinstance(algorithm, Episodic):
        #     self._train_episodic()
        # elif isinstance(algorithm, DynamicProgramming):
        #     self._train_dynamic_programming()
        # else:
        #     raise NotImplementedError

        if settings.algorithm_parameters.derive_v_from_q_as_final_step:
            assert isinstance(algorithm, TabularAlgorithm)
            algorithm.derive_v_from_q()

        if return_result:
            return self._get_result(settings.result_parameters)

    def _train_episodic(self):
        settings = self.settings
        if (settings.review_every_step or settings.display_every_step) and self._agent.set_step_callback:
            self._agent.set_step_callback(self.step)
        print(f"{settings.algorithm_title}: {settings.runs} runs")

        self.max_cum_timestep = 0
        if settings.runs_multiprocessing == common.ParallelContextType.NONE \
                or multiprocessing.current_process().daemon:
            # train in serial
            for run_counter in range(1, settings.runs + 1):
                self.do_run(run_counter)
                self.max_cum_timestep = max(self.max_cum_timestep, self.cum_timestep)
        else:
            self.parallel_runner = ParallelRunner(self)
            self.parallel_runner.do_runs()

        if self._verbose:
            self._agent.print_statistics()

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

        self._agent.algorithm.initialize()

        self.cum_timestep = 0
        # TODO: optionally generate and pre-process episodes in parallel e.g. for Blackjack
        #  according to a behavioural policy which then might change so perhaps in batches
        for episode_counter in range(1, settings.training_episodes + 1):
            self._do_episode(episode_counter)

        if result_parameters:
            return self._get_result(result_parameters)

    def _do_episode(self, episode_counter: int):
        # for use by Breakdown
        self.episode_counter = episode_counter

        settings = self.settings
        if self._verbose or episode_counter % settings.episode_print_frequency == 0:
            print(f"episode_counter = {episode_counter}")

        if not settings.review_every_step and self.cum_timestep != 0:
            self.cum_timestep += 1  # start next episode from the next timestep

        self._agent.parameter_changes(episode_counter)

        algorithm: GeneralAlgorithm = self._agent.algorithm
        assert isinstance(algorithm, Episodic)
        algorithm.do_episode(settings.episode_length_timeout)
        episode = self._agent.episode

        if self._verbose:
            max_t = episode.max_t
            total_return = episode.total_return
            if self._verbose:
                print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
        if not settings.review_every_step:
            self.cum_timestep += episode.max_t
            if self._breakdown:
                self._breakdown.review()

    def _train_dynamic_programming(self):
        settings = self.settings
        algorithm: GeneralAlgorithm = self._agent.algorithm
        assert isinstance(algorithm, DynamicProgramming)

        if settings.review_every_step or settings.display_every_step:
            algorithm.set_step_callback(self.step)
        algorithm.initialize()
        algorithm.run()

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
        result: common.Result = common.Result()

        rp: common.ResultParameters = result_parameters

        if rp.return_recorder and self._breakdown:
            result.recorder = self._breakdown.recorder

        if rp.return_algorithm_title:
            result.algorithm_title = self._agent.algorithm.title

        if isinstance(self._agent, TabularAgent):
            if rp.return_policy_vector:
                result.policy_vector = self._agent.target_policy.get_policy_vector()

            if rp.return_v_vector and self._agent.algorithm.V:
                result.v_vector = self._agent.algorithm.V.vector

            if rp.return_q_matrix and self._agent.algorithm.Q:
                result.q_matrix = self._agent.algorithm.Q.matrix

        if rp.return_cum_timestep:
            result.cum_timestep = self.cum_timestep

        return result
