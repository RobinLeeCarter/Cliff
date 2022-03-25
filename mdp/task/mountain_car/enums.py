import enum
from mdp.model.non_tabular.environment.dimension.dim_enum import DimEnum


class Dim(DimEnum):
    """enumeration of mountain car state and action dimensions"""
    POSITION = enum.auto()
    VELOCITY = enum.auto()
    ACCELERATION = enum.auto()
