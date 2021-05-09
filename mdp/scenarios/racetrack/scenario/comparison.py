from __future__ import annotations
import dataclasses

from mdp.common import utils
from mdp.common.dataclass import comparison

from mdp.scenarios.racetrack.model import environment_parameters


@dataclasses.dataclass
class Comparison(comparison.Comparison):
    # just what is different
    environment_parameters: environment_parameters.EnvironmentParameters = \
        dataclasses.field(default_factory=environment_parameters.default_factory)

    def __post_init__(self):
        super().__post_init__()
        # Push comparison values or default values into most settings attributes if currently =None
        utils.set_none_to_default(self.environment_parameters, environment_parameters.default)
