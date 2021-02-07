from typing import Optional

import utils
import common
import policy
import agent
import algorithm
import train
import view
import environment
import comparison


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.environment = environment.Cliff
        # self.environment.verbose = True
        self.greedy_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.e_greedy_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
                                                                          greedy_policy=self.greedy_policy)
        self.agent = agent.Agent(self.environment, self.e_greedy_policy)

        self.algorithm_factory: algorithm.Factory = algorithm.Factory(self.environment, self.agent)

        # self.comparison_type: Optional[common.ComparisonType] = None
        self.comparison: Optional[comparison.Comparison] = None
        # self.settings_list: list[algorithm.Settings] = []
        # self.algorithms: list[algorithm.EpisodicAlgorithm] = []
        # self.recorder: Optional[train.Recorder] = None
        self.trainer: Optional[train.Trainer] = None

        self.grid_view = view.GridView(self.environment.grid_world)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

        # self.algorithm_: algorithm.EpisodicAlgorithm = algorithm.Sarsa(
        #         self.environment,
        #         self.agent,
        #         alpha=constants.ALPHA,
        #         verbose=False
        #     )

    def setup_and_run(self, comparison_type: common.ComparisonType):
        # self.comparison_type = comparison_type
        # self.settings_list: list[algorithm.Settings] = []
        # recorder_key_type: type
        if comparison_type == common.ComparisonType.RETURN_BY_EPISODE:
            self.comparison = comparison.ReturnByEpisode()
            # self.settings_list = data.return_by_episode_settings
            # recorder_key_type = tuple[algorithm.EpisodicAlgorithm, int]
        elif comparison_type == common.ComparisonType.RETURN_BY_ALPHA:
            self.comparison = comparison.ReturnByAlpha()
            # self.settings_list = self.alpha_settings_list()
            # recorder_key_type = tuple[type, float]
        else:
            raise NotImplementedError
        self.comparison.build()

        # self.algorithms = [self.algorithm_factory[settings_] for settings_ in self.settings_list]
        # self.recorder: train.Recorder = train.Recorder[recorder_key_type]()
        self.trainer = train.Trainer(
            self.algorithm_factory,
            self.comparison,
            # self.recorder,
            verbose=False
        )
        # produce output in self.recorder
        # for settings in self.settings_list:
        #     self.trainer.train(settings)

    # def alpha_settings_list(self) -> list[algorithm.Settings]:
    #     settings_list: list[algorithm.Settings] = []
    #     for alpha in data.alpha_list:
    #         for algorithm_type in data.algorithm_type_list:
    #             settings = algorithm.Settings(
    #                 algorithm_type=algorithm_type,
    #                 parameters={"alpha": alpha}
    #             )
    #             settings_list.append(settings)
    #     return settings_list

    # def compile_return_by_alpha(self):
    #     # collate output from self.recorder
    #     alpha_array = np.array(data.alpha_list, dtype=float)
    #     graph_series: list[view.Series] = []
    #     for algorithm_type in data.algorithm_type_list:
    #         values = np.array([self.recorder[algorithm_type, alpha] for alpha in alpha_array])
    #         series = view.Series(title=algorithm_type.name, values=values)
    #         graph_series.append(series)
    #     self.graph.make_plot(alpha_array, graph_series, is_moving_average=False)

    def run(self):
        # algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray] = {}

        timer: utils.Timer = utils.Timer()
        timer.start()
        for settings in self.comparison.settings_list:
            self.trainer.train(settings)
            timer.lap(name=str(settings.algorithm_title))
            # algorithms_output[algorithm_] = self.trainer.return_array

        timer.stop()

        self.comparison.compile()
        self.comparison.draw_graph()

        # self.behaviour_agent.set_policy(self.target_policy)
        # self.grid_view.open_window()
        # self.view.display_and_wait()
        # self.environment.verbose = True
        # self.agent.verbose = True

        # self.agent.verbose = True
        # self.agent.set_policy(self.greedy_policy)

        # self.grid_view.open_window()
        #
        # recorder_ = recorder.Recorder[str]()
        #
        # test: str = "test"
        #
        # running_average = 0
        # count = 0
        # while count < 100:
        #     self.agent.generate_episode()
        #     episode = self.agent.episode
        #     print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
        #     count += 1
        #     running_average += (1/count) * (episode.total_return - running_average)
        #     print(f"count: {count} \t running_average: {running_average:.1f}")
        #     recorder_[test] = episode.total_return
        #     value = recorder_[test]
        #     print(f"recorder value: {value:.1f}")
        # user_event: common.UserEvent = self.grid_view.display_episode(episode, show_trail=False)
        # if user_event == common.UserEvent.QUIT:
        #     break
