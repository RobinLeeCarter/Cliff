from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

from comparison_dataclasses import series


class Graph:
    def __init__(self):
        self.title = ""
        self.fig: Optional[figure.Figure] = None
        self.ax: Optional[figure.Axes] = None

    def make_plot(self,
                  x_series: series.Series,
                  x_min: float,
                  x_max: float,
                  y_min: float,
                  y_max: float,
                  graph_series: list[series.Series],
                  moving_average_window_size: int = 0):
        is_moving_average = (moving_average_window_size >= 3)
        if is_moving_average:
            self.title = f"Moving average of Average Return vs {x_series.title}"
        else:
            self.title = f"Average Return vs {x_series.title}"

        self.prep_graph(x_min=x_min, x_max=x_max, x_label=x_series.title,
                        y_min=y_min, y_max=y_max)
        if is_moving_average:
            self.moving_average_plot(x_series, graph_series, moving_average_window_size)
        else:
            self.plot_arrays(x_series, graph_series)
        self.ax.legend()
        plt.show()

    def prep_graph(self, x_min: float, x_max: float, x_label: str, y_min: float, y_max: float):
        self.fig: figure.Figure = plt.figure()
        self.ax: figure.Axes = self.fig.subplots()
        self.ax.set_title(self.title)
        self.ax.set_xlim(xmin=x_min, xmax=x_max)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylim(ymin=y_min, ymax=y_max)
        self.ax.set_ylabel("Average Return")
        self.ax.grid(True)

    def plot_arrays(self, x_series: series.Series, graph_series: list[series.Series]):
        for series_ in graph_series:
            self.ax.plot(x_series.values, series_.values, label=series_.title)

    def moving_average_plot(self,
                            x_series: series.Series,
                            graph_series: list[series.Series],
                            moving_average_window_size: int):
        # convert output into moving averages
        graph_series_ma: list[series.Series] = graph_series.copy()
        for series_ma in graph_series_ma:
            values_ma = self.moving_average(series_ma.values, window_size=moving_average_window_size)
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
