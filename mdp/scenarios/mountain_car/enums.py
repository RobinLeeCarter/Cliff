import enum
from mdp.model.environment.non_tabular.dimension.dim_enum import DimEnum


class Dim(DimEnum):
    """enumeration of mountain car state and action dimensions"""
    POSITION = enum.auto()
    VELOCITY = enum.auto()
    ACCELERATION = enum.auto()
