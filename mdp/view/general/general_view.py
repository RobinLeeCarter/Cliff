from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp import common
    from mdp.general_controller import GeneralController

from mdp.view.general.graph2d import Graph2D
from mdp.view.general.graph3d import Graph3D


class GeneralView(ABC):
    def __init__(self):
        self._controller: Optional[GeneralController] = None
        self._comparison: Optional[common.Comparison] = None
        self.graph2d: Optional[Graph2D] = None
        self.graph3d: Optional[Graph3D] = None

    def set_controller(self, controller: GeneralController):
        self._controller: GeneralController = controller

    def build(self, comparison: common.Comparison):
        """build top-down"""
        self._comparison = comparison
        self.graph2d: Graph2D = self._create_graph()
        self.graph3d: Graph3D = self._create_graph3d()

    def _create_graph(self) -> Graph2D:
        return Graph2D()

    def _create_graph3d(self) -> Graph3D:
        return Graph3D()

    def demonstrate(self):
        raise Exception("demonstrate() not implemented")
