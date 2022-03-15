from __future__ import annotations
from dataclasses import dataclass, field

import utils
from mdp import common

from mdp.scenario.cliff.model import environment_parameters


@dataclass
class Comparison(common.Comparison):
    # just what is different
    environment_parameters: environment_parameters.EnvironmentParameters = \
        field(default_factory=environment_parameters.default_factory)

    def __post_init__(self):
        super().__post_init__()
        # Push comparison values or default values into most settings attributes if currently =None
        utils.set_none_to_default(self.environment_parameters, environment_parameters.default)
