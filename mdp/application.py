from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.general.general_model import GeneralModel
    from mdp.view.general.general_view import GeneralView
    from mdp.controller.general_controller import GeneralController
from mdp.scenario.comparison_factory import ComparisonFactory
from mdp.mvc_factory import MVCFactory


class Application:
    def __init__(self, comparison_type: Optional[common.ComparisonType] = None):
        self._comparison_factory: ComparisonFactory = ComparisonFactory()
        self._comparison: Optional[common.Comparison] = None
        self._mvc_factory: MVCFactory = MVCFactory()
        self.model: Optional[GeneralModel] = None
        self.view: Optional[GeneralView] = None
        self.controller: Optional[GeneralController] = None
        if comparison_type:
            self._comparison: common.Comparison = self._comparison_factory.create(comparison_type)
            self.build(self._comparison)

    def build(self, comparison: common.Comparison):
        self._comparison = comparison
        environment_type: common.EnvironmentType = comparison.environment_parameters.environment_type
        self.model, self.view, self.controller = self._mvc_factory.create(environment_type)
        self.controller.link_mvc(self.model, self.view)
        self.controller.build(comparison)

    def run(self):
        self.controller.run()
        self.controller.output()
