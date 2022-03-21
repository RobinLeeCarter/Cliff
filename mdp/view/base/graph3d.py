from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, cm

if TYPE_CHECKING:
    from mdp import common


class Graph3D:
    def __init__(self):
        self.title = ""
        self.fig: Optional[figure.Figure] = None
        self.ax: Optional[figure.Axes] = None
        self.graph3d_values: Optional[common.Graph3DValues] = None

    def make_plot(self, graph_values: common.Graph3DValues):
        self.graph3d_values = graph_values

        # fill-in values where missing
        gv = self.graph3d_values
        if gv.x_label is None:
            gv.x_label = gv.x_series.title
        if gv.y_label is None:
            gv.y_label = gv.y_series.title
        if gv.z_label is None:
            gv.z_label = gv.z_series.title
        if gv.title is None:
            gv.title = f"{gv.z_label} by {gv.x_label} and {gv.y_label}"

        self.pre_plot()
        self.plot(gv.x_series, gv.y_series, gv.z_series)
        self.post_plot()
        plt.show()

    def pre_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # self.ax = self.fig.subplots(projection='3d')          Fails

    def plot(self, x_series: common.Series, y_series: common.Series, z_series: common.Series):
        x_grid, y_grid = np.meshgrid(x_series.values, y_series.values)
        z_values = z_series.values
        # noinspection PyUnresolvedReferences
        cmap = cm.coolwarm
        self.ax.plot_surface(x_grid, y_grid, z_values, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)

    def post_plot(self):
        gv = self.graph3d_values
        self.ax.set_title(gv.title)
        self.ax.set_xlim(xmin=gv.x_min, xmax=gv.x_max)
        self.ax.set_xlabel(gv.x_label)
        self.ax.set_ylim(ymin=gv.y_min, ymax=gv.y_max)
        self.ax.set_ylabel(gv.y_label)
        self.ax.set_zlim(zmin=gv.z_min, zmax=gv.z_max)
        self.ax.set_zlabel(gv.z_label)
        if gv.has_grid:
            self.ax.grid(True)
        if gv.has_legend:
            self.ax.legend()


