from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from mdp.model.agent.agent import Agent
    from mdp.model.agent.episode import Episode
    from mdp.model.breakdown.breakdown import Breakdown
from mdp import common
from mdp.model.algorithm.abstract.algorithm import Algorithm
from mdp.model.algorithm.abstract.episodic import Episodic
from mdp.model.algorithm.abstract.dynamic_programming import DynamicProgramming
from mdp.model.trainer.parallel_runner import ParallelRunner


class Trainer:
    def __init__(self,
                 agent_: Agent,
                 breakdown_: Optional[Breakdown],
                 model_step_callback: Optional[Callable[[Optional[Episode]], None]] = None,
                 verbose: bool = False
                 ):
        self._agent: Agent = agent_
        self._breakdown: Optional[Breakdown] = breakdown_
        self._algorithm: Optional[Algorithm] = None
        self.settings: Optional[common.Settings] = None

        self._model_step_callback: Optional[Callable[[Optional[Episode]], None]] = model_step_callback
        self._verbose = verbose
        self._cont: bool = True

        self.episode_counter: int = 0

        self.cum_timestep: int = 0  # cumulative timestep across all episodes
        self.max_cum_timestep: int = 0  # max timestep across all runs

        self._result: Optional[common.Result] = None

    @property
    def breakdown(self) -> Breakdown:
        return self._breakdown

    @property
    def episode(self) -> Episode:
        return self._agent.episode

    @property
    def agent(self) -> Agent:
        return self._agent

    def disable_step_callback(self):
        self._model_step_callback = None

    def train(self, settings: common.Settings) -> common.Result:
        # process settings
        self.settings = settings
        self._agent.apply_settings(self.settings)
        algorithm: Algorithm = self._agent.algorithm
        settings.algorithm_title = algorithm.title      # moved from episodic

        if isinstance(algorithm, Episodic):
            self._train_episodic(settings, algorithm)
        elif isinstance(algorithm, DynamicProgramming):
            self._train_dynamic_programming(settings, algorithm)
        else:
            raise NotImplementedError

        if settings.algorithm_parameters.derive_v_from_q_as_final_step:
            algorithm.derive_v_from_q()

        self._build_result(settings.result_parameters)
        return self._result

    def _train_episodic(self, settings: common.Settings, algorithm: Episodic):
        if (settings.review_every_step or settings.display_every_step) and self._agent.set_step_callback:
            self._agent.set_step_callback(self.step)
        print(f"{settings.algorithm_title}: {settings.runs} runs")

        self.max_cum_timestep = 0

        # TODO: run this in parallel as an option (separate recorders again that get combined?)
        if settings.runs_multiprocessing == common.ParallelContextType.NONE:
            for run_counter in range(1, settings.runs + 1):
                # needs to return max_cum_timestep
                self.do_run(settings, run_counter)
        else:
            self.parallel_runner = ParallelRunner(self)
            self.parallel_runner.do_runs()

        if self._verbose:
            self._agent.print_statistics()

    # TODO: return result for parallel runs somehow
    def do_run(self, settings: common.Settings, run_counter: int):
        if self._verbose or run_counter % settings.run_print_frequency == 0:
            print(f"run_counter = {run_counter}: {settings.training_episodes} episodes")

        algorithm = self._agent.algorithm
        algorithm.initialize()

        self.cum_timestep = 0
        # self.episode_counter is used in breakdown :(
        for self.episode_counter in range(1, settings.training_episodes + 1):
            self._do_episode(settings, algorithm, self.episode_counter)

        self.max_cum_timestep = max(self.max_cum_timestep, self.cum_timestep)
        # build result and return it?

    def _do_episode(self, settings: common.Settings, algorithm: Episodic, episode_counter: int):
        if self._verbose or episode_counter % settings.episode_print_frequency == 0:
            print(f"episode_counter = {episode_counter}")

        if not settings.review_every_step and self.cum_timestep != 0:
            self.cum_timestep += 1  # start next episode from the next timestep

        self._agent.parameter_changes(episode_counter)
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

    # noinspection PyUnusedLocal
    def _train_dynamic_programming(self, settings: common.Settings, algorithm_: DynamicProgramming):
        if (settings.review_every_step or settings.display_every_step) and algorithm_.set_step_callback:
            algorithm_.set_step_callback(self.step)
        algorithm_.initialize()
        algorithm_.run()

    def step(self) -> bool:
        if self.settings.review_every_step:
            self._review_step()
        if self.settings.display_every_step and self._model_step_callback:
            self._model_step_callback(self._agent.episode)
        return True

    def _review_step(self):
        self.cum_timestep += 1
        self._breakdown.review()

    def _build_result(self, result_parameters: common.ResultParameters):
        self._result = common.Result()

        rp: common.ResultParameters = result_parameters
        if rp.return_algorithm_title:
            self._result.algorithm_title = self._agent.algorithm.title
        if rp.return_recorder and self._breakdown:
            self._result.recorder = self._breakdown.recorder
        if rp.return_policy_vector:
            self._result.policy_vector = self._agent.target_policy.get_policy_vector()
        if rp.return_v_vector and self._agent.algorithm.V:
            self._result.v_vector = self._agent.algorithm.V.vector
        if rp.return_q_matrix and self._agent.algorithm.Q:
            self._result.q_matrix = self._agent.algorithm.Q.matrix
