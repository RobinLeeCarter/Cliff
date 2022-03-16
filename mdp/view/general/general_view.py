from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.controller.general_controller import GeneralController

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
        self.graph2d: Graph2D = Graph2D()
        self.graph3d: Graph3D = Graph3D()

    def demonstrate(self):
        raise Exception("demonstrate() not implemented")
