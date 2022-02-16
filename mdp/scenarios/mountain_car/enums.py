import enum
from mdp.model.environment.non_tabular.dimension_enum import DimensionEnum


@enum.unique
class StateFloatDim(DimensionEnum):
    POSITION = enum.auto()
    VELOCITY = enum.auto()


# @enum.unique
# class StateCategoryDim(DimensionEnum):
#     pass


@enum.unique
class ActionCategoryDim(DimensionEnum):
    ACCELERATION = enum.auto()
