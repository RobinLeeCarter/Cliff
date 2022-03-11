from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

if TYPE_CHECKING:
    from mdp import common


class Graph2D:
    def __init__(self):
        self.title = ""
        self.fig: Optional[figure.Figure] = None
        self.ax: Optional[figure.Axes] = None
        self.graph_values: Optional[common.GraphValues] = None

    def make_plot(self, graph_values: common.GraphValues):
        self.graph_values = graph_values

        # fill-in values where missing
        gv = self.graph_values
        if gv.x_label is None:
            gv.x_label = gv.x_series.title
        if gv.title is None:
            vs = f"{gv.y_label} vs {gv.x_label}"
            if gv.moving_average_window_size is None:
                gv.title = vs
            else:
                self.title = f"Moving average of {vs}"

        self.pre_plot()
        if gv.moving_average_window_size is None:
            self.plot(gv.x_series, gv.graph_series)
        else:
            self.moving_average_plot(gv.x_series, gv.graph_series, gv.moving_average_window_size)
        self.post_plot()
        plt.show()

    def pre_plot(self):
        self.fig: figure.Figure = plt.figure()
        self.ax: figure.Axes = self.fig.subplots()

    def plot(self, x_series: common.Series, graph_series: list[common.Series]):
        for series_ in graph_series:
            self.ax.plot(x_series.values, series_.values, label=series_.title)

    def post_plot(self):
        gv = self.graph_values
        self.ax.set_title(gv.title)
        self.ax.set_xlim(xmin=gv.x_min, xmax=gv.x_max)
        self.ax.set_xlabel(gv.x_label)
        self.ax.set_ylim(ymin=gv.y_min, ymax=gv.y_max)
        self.ax.set_ylabel(gv.y_label)
        if gv.has_grid:
            self.ax.grid(True)
        if gv.has_legend:
            self.ax.legend()

    def moving_average_plot(self,
                            x_series: common.Series,
                            graph_series: list[common.Series],
                            moving_average_window_size: int):
        # convert output into moving averages
        graph_series_ma: list[common.Series] = graph_series.copy()
        for series_ma in graph_series_ma:
            values_ma = self.moving_average(series_ma.values, window_size=moving_average_window_size)
            series_ma.values = values_ma
        self.plot(x_series, graph_series_ma)

    def moving_average(self, values: np.ndarray, window_size: int) -> np.ndarray:
        cum_sum = np.cumsum(np.insert(values, 0, 0))
        mov_av = (cum_sum[window_size:] - cum_sum[:-window_size]) / window_size
        full_ma = np.empty(shape=values.shape, dtype=float)
        full_ma.fill(np.NAN)
        nan_indent: int = int((window_size - 1) / 2)
        full_ma[nan_indent:-nan_indent] = mov_av
        return full_ma
