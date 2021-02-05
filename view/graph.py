from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import constants
from data import series


class Graph:
    def __init__(self):
        self.title = ""
        self.fig: Optional[figure.Figure] = None
        self.ax: Optional[figure.Axes] = None

    def make_plot(self,
                  x_series: np.ndarray,
                  graph_series: list[series.Series],
                  is_moving_average: bool = False):
        if is_moving_average:
            self.title = "Moving average of Average Return vs Learning Episodes"
        else:
            self.title = "Average Return vs Learning Episodes"

        self.prep_graph()
        if is_moving_average:
            self.moving_average_plot(x_series, graph_series)
        else:
            self.plot_arrays(x_series, graph_series)
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

    def plot_arrays(self, x_series: np.ndarray, graph_series: list[series.Series]):
        for series_ in graph_series:
            self.ax.plot(x_series, series_.values, label=series_.title)

    def moving_average_plot(self,
                            x_series: np.ndarray,
                            graph_series: list[series.Series]):
        # convert output into moving averages
        graph_series_ma: list[series.Series] = graph_series.copy()
        for series_ma in graph_series_ma:
            values_ma = self.moving_average(series_ma.values, window_size=constants.MOVING_AVERAGE_WINDOW_SIZE)
            series_ma.values = values_ma
        self.plot_arrays(x_series, graph_series_ma)

    def moving_average(self, values: np.ndarray, window_size: int) -> np.ndarray:
        cum_sum = np.cumsum(np.insert(values, 0, 0))
        mov_av = (cum_sum[window_size:] - cum_sum[:-window_size]) / window_size
        full_ma = np.empty(shape=values.shape, dtype=float)
        full_ma.fill(np.NAN)
        nan_indent: int = int((window_size - 1) / 2)
        full_ma[nan_indent:-nan_indent] = mov_av
        return full_ma
