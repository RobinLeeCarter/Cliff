import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import constants
import algorithm


class Graph:
    def __init__(self):
        pass

    def simple_plot(self, iteration: np.ndarray, algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray]):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_title("Average Return vs Learning Episodes")
        ax.set_xlim(xmin=0, xmax=constants.TRAINING_ITERATIONS)
        ax.set_xlabel("Learning Episodes")
        ax.set_ylim(ymin=-100, ymax=0)
        ax.set_ylabel("Average Return")
        for algorithm_, return_array in algorithms_output.items():
            ax.plot(iteration, return_array, label=algorithm_.title)
        ax.legend()
        ax.grid(True)
        plt.show()

    def ma_plot(self, iteration: np.ndarray, algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray]):
        algorithms_ma: dict[algorithm.EpisodicAlgorithm, np.ndarray] = {}
        for algorithm_, return_array in algorithms_output.items():
            algorithms_ma[algorithm_] = self.moving_average(return_array,
                                                            window_size=constants.MOVING_AVERAGE_WINDOW_SIZE)

        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_title("Moving average of Average Return vs Learning Episodes")
        ax.set_xlim(xmin=0, xmax=constants.TRAINING_ITERATIONS)
        ax.set_xlabel("Learning Episodes")
        ax.set_ylim(ymin=-100, ymax=0)
        ax.set_ylabel("Average Return")
        for algorithm_, return_array in algorithms_ma.items():
            ax.plot(iteration, return_array, label=algorithm_.title)
        ax.legend()
        ax.grid(True)
        plt.show()

    def moving_average(self, series: np.ndarray, window_size: int) -> np.ndarray:
        cum_sum = np.cumsum(np.insert(series, 0, 0))
        mov_av = (cum_sum[window_size:] - cum_sum[:-window_size]) / window_size
        full_ma = np.empty(shape=series.shape, dtype=float)
        full_ma.fill(np.NAN)
        nan_indent: int = int((window_size - 1) / 2)
        full_ma[nan_indent:-nan_indent] = mov_av
        return full_ma
