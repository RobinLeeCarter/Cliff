from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import constants
import algorithm


class Graph:
    def __init__(self):
        self.title = ""
        self.fig: Optional[figure.Figure] = None
        self.ax: Optional[figure.Axes] = None

    def make_plot(self,
                  iteration: np.ndarray,
                  algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray],
                  is_moving_average: bool = False):
        if is_moving_average:
            self.title = "Moving average of Average Return vs Learning Episodes"
        else:
            self.title = "Average Return vs Learning Episodes"

        self.prep_graph()
        if is_moving_average:
            self.moving_average_plot(iteration, algorithms_output)
        else:
            self.plot_arrays(iteration, algorithms_output)
        self.ax.legend()
        plt.show()

    def prep_graph(self):
        self.fig: figure.Figure = plt.figure()
        self.ax: figure.Axes = self.fig.subplots()
        self.ax.set_title(self.title)
        self.ax.set_xlim(xmin=0, xmax=constants.TRAINING_ITERATIONS)
        self.ax.set_xlabel("Learning Episodes")
        self.ax.set_ylim(ymin=-100, ymax=0)
        self.ax.set_ylabel("Average Return")
        self.ax.grid(True)

    def plot_arrays(self, iteration: np.ndarray, algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray]):
        for algorithm_, return_array in algorithms_output.items():
            self.ax.plot(iteration, return_array, label=algorithm_.title)

    def moving_average_plot(self,
                            iteration: np.ndarray,
                            algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray]):
        # convert output into moving averages
        algorithms_ma: dict[algorithm.EpisodicAlgorithm, np.ndarray] = {}
        for algorithm_, return_array in algorithms_output.items():
            algorithms_ma[algorithm_] = self.moving_average(return_array,
                                                            window_size=constants.MOVING_AVERAGE_WINDOW_SIZE)
        self.plot_arrays(iteration, algorithms_ma)

    def moving_average(self, series: np.ndarray, window_size: int) -> np.ndarray:
        cum_sum = np.cumsum(np.insert(series, 0, 0))
        mov_av = (cum_sum[window_size:] - cum_sum[:-window_size]) / window_size
        full_ma = np.empty(shape=series.shape, dtype=float)
        full_ma.fill(np.NAN)
        nan_indent: int = int((window_size - 1) / 2)
        full_ma[nan_indent:-nan_indent] = mov_av
        return full_ma
