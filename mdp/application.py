from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.general.general_model import GeneralModel
    from mdp.view.general.general_view import GeneralView
    from mdp.controller.general_controller import GeneralController
from mdp.scenario.comparison_factory import ComparisonFactory
from mdp.mvc_factory import MVCFactory


class Application:
    def __init__(self, comparison_type: common.ComparisonType):
        self._comparison_factory: ComparisonFactory = ComparisonFactory()
        self._mvc_factory: MVCFactory = MVCFactory()

        self._comparison: common.Comparison = self._comparison_factory.create(comparison_type)

        self._model: GeneralModel
        self._view: GeneralView
        self._controller: GeneralController
        environment_type: common.EnvironmentType = self._comparison.environment_parameters.environment_type
        self._model, self._view, self._controller = self._mvc_factory.create(environment_type)
        self._controller.link_mvc(self._model, self._view)

        self._controller.build(self._comparison)
        self._controller.run()
        self._controller.output()
