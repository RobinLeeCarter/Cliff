from __future__ import annotations

from typing import Optional, Callable, TYPE_CHECKING, TypeVar

import numpy as np

from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dim_enum import DimEnum
    from mdp.model.non_tabular.environment.dimension.dims import Dims
    from mdp.model.non_tabular.feature.tile_coding.tiling_coding_parameters import TileCodingParameters
from mdp import common
from mdp.model.non_tabular.feature.tile_coding.tiling_group import TilingGroup
from mdp.model.non_tabular.feature.sparse_feature import SparseFeature
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class TileCoding(SparseFeature[State, Action],
                 feature_type=common.FeatureType.TILE_CODING):
    def __init__(self, dims: Dims, tile_coding_parameters: TileCodingParameters):
        """
        :param dims: the dimensions of the space being covered and whether continuous or categorical
        """
        super().__init__(dims, tile_coding_parameters)
        # max_size: optional: maximum tile number, will reuse from start if using a dict and show a warning

        # self._state_float_dimensions: dict[DimEnum, FloatDimension] = state_float_dimensions
        # self._state_category_dimensions: dict[DimEnum, CategoryDimension] = state_category_dimensions
        # self._action_category_dimensions: dict[DimEnum, CategoryDimension] = action_category_dimensions
        # self._max_size: Optional[int] = max_size
        self._use_dict: bool = tile_coding_parameters.use_dict

        self._tiling_groups: list[TilingGroup] = []
        self._total_tilings: int = 0
        self._total_tiles: int = 0
        self._reached_max_size: bool = False

        # state and action converted into tuples
        # this is stored for states so that when all actions are needed the state part is only done once
        # one whole dictionary for each state
        # tiling_group_index, tiling -> tile_coord + category_index
        self._state_tuples: dict[[int, int], tuple] = {}
        self._action_tuples: dict[int, tuple] = {}

        # dictionary of coord to tile_indexes
        self._tile_dict: dict[tuple, int] = {}
        # counter of tile_indexes for dictionary
        self._tile_index_counter: int = 0
        self._collision_warning: bool = False

        for tiling_group_parameters in tile_coding_parameters.tiling_groups:
            self._add(tiling_group_parameters)

    @property
    def use_dict(self) -> bool:
        return self._use_dict

    def add(self,
            included_dims: set[DimEnum],
            tile_size_per_dim: Optional[dict[DimEnum, float]] = None,
            tiles_per_dim: Optional[dict[DimEnum, int]] = None,
            tilings: Optional[int] = None,
            offset_per_dimension_fn: Optional[Callable[[int], np.ndarray]] = None
            ):
        """
        :param included_dims: unordered set of dimensions to include in the tile coding group being added
        :param tile_size_per_dim: either specify tile_size in each dimension (distance over which to generalise)...
        :param tiles_per_dim: or specify numbers of tiles in each dimension (all dimensions not just included ones)
        :param tilings: number of tilings e.g. n=8, should be a power of 2 and >= 4k (k = number of included dimensions)
        if unspecified or None, will use the lowest power of 2 above 4k
        :param offset_per_dimension_fn: fn that returns offset units in each dimension given the number of dimensions
        """
        tiling_group_parameters: TilingGroupParameters = TilingGroupParameters(
            included_dims=included_dims,
            tile_size_per_dim=tile_size_per_dim,
            tiles_per_dim=tiles_per_dim,
            tilings=tilings,
            offset_per_dimension_fn=offset_per_dimension_fn
        )
        self._add(tiling_group_parameters)

    def _add(self, tiling_group_parameters: TilingGroupParameters):
        tiling_group = TilingGroup(self._dims,
                                   tiling_group_parameters)
        self._tiling_groups.append(tiling_group)
        self._total_tilings += tiling_group.tilings
        if self._use_dict:
            self._total_tiles += tiling_group.total_tiles
            # TODO: Is this correct?
            self._max_size = self._total_tiles
            # if self._size != 0 and self._total_tiles > self._size:
            #     print(f"Warning: total_tiles: {self._total_tiles} exceeds size: {self._size}")

    def _do_state_computation(self):
        for tiling_group_index, tiling_group in enumerate(self._tiling_groups):
            # array of: tilings x included dimensions
            float_tile_coords = tiling_group.get_float_tile_coords(self._state.floats())
            # array of included dimensions in each case
            state_categories: np.ndarray = tiling_group.filter_state_categories(self._state.categories())
            for tiling, tile_coord in enumerate(float_tile_coords):
                state_tuple: tuple = tuple(tile_coord) + tuple(state_categories)
                self._state_tuples[tiling_group_index, tiling] = state_tuple

    def _do_action_computation(self):
        for tiling_group_index, tiling_group in enumerate(self._tiling_groups):
            action_categories: np.ndarray = tiling_group.filter_action_categories(self._action.categories())
            self._action_tuples[tiling_group_index] = tuple(action_categories)

    def _get_sparse_vector(self) -> np.ndarray:
        """return just the indexes of x which are 1 (rest are 0) (i.e. a list of tiles) using unpacked values"""
        tile_indexes: list[int] = []
        for tiling_group_index, tiling_group in enumerate(self._tiling_groups):
            for tiling in range(tiling_group.tilings):
                full_tuple = (tiling_group_index, tiling) + self._state_tuples[tiling_group_index, tiling]
                if self._action:
                    full_tuple += self._action_tuples[tiling_group_index]
                tile_index: int = self._get_tile_index(full_tuple)
                tile_indexes.append(tile_index)
        return np.array(tile_indexes, dtype=np.int)

    # def _get_x_sparse(self) -> np.ndarray:
    #     """return just the indexes of x which are 1 (rest are 0) (i.e. a list of tiles) using unpacked values"""
    #     tile_indexes: list[int] = []
    #     for tiling_group_index, tiling_group in enumerate(self._tiling_groups):
    #         # array of: tilings x included dimensions
    #         float_tile_coords = tiling_group.get_float_tile_coords(self._state_floats)
    #         # array of included dimensions in each case
    #         state_categories: np.ndarray = tiling_group.filter_state_categories(self._state_categories)
    #         action_categories: np.ndarray = tiling_group.filter_action_categories(self._action_categories)
    #         for tiling, tile_coord in enumerate(float_tile_coords):
    #             state_tuple = (tiling_group_index, tiling) + tuple(tile_coord) + tuple(state_categories)
    #             full_tuple: tuple = state_tuple + tuple(action_categories)
    #             tile_index: int = self._get_tile_index(full_tuple)
    #             tile_indexes.append(tile_index)
    #     return np.array(tile_indexes, dtype=np.int)

    # def __getitem__(self, state_values: tuple[float, ...]) -> np.ndarray:
    #     """
    #     :param state_values: full state values as np.ndarray
    #     :return: np.ndarray of active tile indexes
    #     """
    #     state = np.array(state_values)
    #     tile_results_list: list[tuple[np.ndarray, np.ndarray]]\
    #         = [tile_coder[state] for tile_coder in self._tiling_groups]
    #     tile_indexes: list[int] = []
    #     for tiling_group_index, (tile_coords, int_values) in enumerate(tile_results_list):
    #         int_tuple = tuple(int_values)
    #         for tiling, coord in enumerate(tile_coords):
    #             full_coord_tuple: tuple[int, ...] = (tiling_group_index, tiling) + tuple(coord) + int_tuple
    #             tile_index: int = self._get_tile_index(full_coord_tuple)
    #             tile_indexes.append(tile_index)
    #     return np.array(tile_indexes, dtype=np.int)

    def _get_tile_index(self, full_tuple: tuple) -> int:
        """
        :param full_tuple: full coord includes tiling_group, tiling, float coords and int coords
        :return: the tile index for the full_coord_tuple
        """
        if self._use_dict:
            # want to get value if present, else set value and increment
            tile_index = self._tile_dict.get(full_tuple)   # avoids raising KeyError
            if tile_index is None:
                if self._reached_max_size:
                    if not self._collision_warning:
                        self._collision_warning = True
                        print(f"Warning: Collisions, switching to hash method because"
                              f" tile dict indexes exceeded output size of {self._max_size}")
                    tile_index = hash(full_tuple) % self._max_size
                else:
                    tile_index = self._tile_index_counter
                    self._tile_dict[full_tuple] = tile_index
                    self._tile_index_counter += 1
                    if self._max_size is not None and self._tile_index_counter >= self._max_size:
                        self._reached_max_size = True
        else:
            tile_index = hash(full_tuple) % self._max_size
        return tile_index

    @property
    def count(self) -> int:
        """
        could be that few tiles and therefore weights are needed as only in a subset of the space is being used
        :return: "The number of indices currently used.
        All used indices will be between 0 and one less than this number.
        The count is always less than the size, and only count have been used.
        Knowing the count can often speed operations that only need be done on the used features
        (e.g., eligibility traces)."
        """
        if self._use_dict:
            if self._reached_max_size:
                return self._max_size
            else:
                return self._tile_index_counter + 1
        else:
            return self._max_size

    @property
    def max_size(self) -> int:
        """
        :return: maximum number of possible tiles can can be returned ever from this tile coding
        """
        if self._use_dict:
            if self._max_size is not None:
                return min(self._total_tiles, self._max_size)
            else:
                return self._total_tiles
        else:
            return self._max_size

    @property
    def tilings(self) -> int:
        """
        :return: total number of tilings from all tiling groups (used to set step-size alpha)
        """
        return self._total_tilings

    def build_complete_dict(self):
        print("build_complete_dict")
        assert self._use_dict
        for tiling_group_index, tiling_group in enumerate(self._tiling_groups):
            tile_coord_list: list[tuple] = tiling_group.get_all_tile_coords()
            for tile_coord in tile_coord_list:
                full_tuple = (tiling_group_index, ) + tile_coord
                self._get_tile_index(full_tuple)
        print(self._tile_index_counter)
        # self._tile_dict[full_tuple] = self._tile_index_counter
        # self._tile_index_counter += 1
        # if self._max_size is not None and self._tile_index_counter >= self._max_size:
        #     self._reached_max_size = True
