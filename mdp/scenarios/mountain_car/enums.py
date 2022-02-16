import enum
from mdp.model.environment.non_tabular.dim_enum import DimEnum


class Dim(DimEnum):
    """enumeration of dimensions just to avoid using magic strings and to be consistent with other code"""
    POSITION = enum.auto()
    VELOCITY = enum.auto()
    ACCELERATION = enum.auto()
