import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import common
import constants
import environment
import policy
import agent
from algorithm import on_policy_td_control
import view


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.environment = environment.Environment(constants.GRID, self.rng, verbose=False)
        self.greedy_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.e_greedy_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
                                                                          greedy_policy=self.greedy_policy)
        self.agent = agent.Agent(self.environment, self.e_greedy_policy)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment, self.rng)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

        self.algorithm_: on_policy_td_control.OnPolicyTdControl = on_policy_td_control.OnPolicyTdControl(
                self.environment,
                self.agent,
                alpha=constants.ALPHA,
                verbose=False
            )
        self.view = view.View(self.environment.grid_world)

    def run(self):
        self.algorithm_.run()
        self.algorithm_.Q.print_coverage_statistics()
        # self.output_q()
        # self.graph_samples(self.algorithm_.sample_iteration, self.algorithm_.average_return)

        # self.behaviour_agent.set_policy(self.target_policy)
        self.view.open_window()
        # self.view.display_and_wait()
        # self.environment.verbose = True
        # self.agent.verbose = True

        # self.agent.verbose = True
        # self.agent.set_policy(self.greedy_policy)
        while True:
            self.agent.generate_episode()
            print(f"max_t: {self.agent.episode.max_t}")
            user_event: common.UserEvent = self.view.display_episode(self.agent.episode, show_trail=False)
            if user_event == common.UserEvent.QUIT:
                break

    # def output_q(self):
    #     q = self.algorithm_.Q
    #     q_size = q.size
    #     q_non_zero = np.count_nonzero(q)
    #     percent_non_zero = 100.0 * q_non_zero / q_size
    #     print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")

    def graph_samples(self, iteration: np.ndarray, average_return: np.ndarray):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_title("Average Return vs Learning Episodes")
        ax.set_xlim(xmin=0, xmax=constants.LEARNING_EPISODES)
        ax.set_xlabel("Learning Episodes")
        ax.set_ylim(ymin=constants.INITIAL_Q_VALUE*2, ymax=0)
        ax.set_ylabel("Average Return")
        ax.plot(iteration, average_return)
        plt.show()
