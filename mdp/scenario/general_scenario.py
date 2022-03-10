from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from mdp.model.general.environment.general_environment import GeneralEnvironment
from mdp import common
from mdp.model.general.general_model import GeneralModel
from mdp.view.general_view import GeneralView
from mdp.general_controller import GeneralController


Model = TypeVar('Model', bound=GeneralModel)
View = TypeVar('View', bound=GeneralView)
Controller = TypeVar('Controller', bound=GeneralController)
# Environment = TypeVar('Environment', bound=GeneralEnvironment)


class GeneralScenario(Generic[Model, View, Controller], ABC):
    def __init__(self, **_ignored):
        # self._comparison_type: common.ComparisonType = comparison_type
        self._model: Model = self._create_model()
        self._view: View = self._create_view()
        self._controller: Controller = self._create_controller()
        self._controller.link_mvc(self._model, self._view)
        self._comparison: Optional[common.Comparison] = None

    def build(self):
        self._comparison: common.Comparison = self._create_comparison()
        # self._comparison = self._get_comparison(self._comparison_type)
        self._controller.build(self._comparison)

    def run(self):
        self._controller.run()
        self._controller.output()

    @abstractmethod
    def _create_model(self) -> Model:
        ...

    @abstractmethod
    def _create_view(self) -> View:
        ...

    @abstractmethod
    def _create_controller(self) -> Controller:
        ...

    @abstractmethod
    def _create_comparison(self) -> common.Comparison:
        ...

    @property
    def environment(self) -> GeneralEnvironment:
        environment = self._model.environment
        if environment is None:
            raise Exception("Environment is None")
        else:
            return self._model.environment

    # @abstractmethod
    # def _get_comparison(self, comparison_type: common.ComparisonType) -> common.Comparison:
    #     pass

