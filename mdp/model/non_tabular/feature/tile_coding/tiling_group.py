from __future__ import annotations

import itertools
import math
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dim_enum import DimEnum
    from mdp.model.non_tabular.environment.dimension.dims import Dims
    from mdp.model.non_tabular.environment.dimension.float_dimension import FloatDimension
    from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters


class TilingGroup:
    """
    Note: may actually need to "hash-down" to just the states that are actually used
    http://www.incompleteideas.net/RL-FAQ.html
    There are at least three common tricks:
    1) you can consider subsets of the state variables in separate tilings,
    2) you can use overlapping tilings to get vastly higher resolution in high dimensional spaces than would be
     possible with a simple grid using the same amount of memory, and
    3) you can hash down to a smaller space so as only to devote memory to the portions
     of state space that actually occur.
    """
    def __init__(self,
                 dims: Dims,
                 tiling_group_parameters: TilingGroupParameters):
        """
        :param dims: structure to holds all the information about the dimensions of the environment
        """
        # :param included_dims: set of the dims to tile or not (based on dims, in any order)
        # For each dimension either specify tile_size or tiles or if leave blank tiles will equal tilings
        # :param tile_size_per_dim: tile_size (distance over which to generalise) of dims (in any order)
        # :param tiles_per_dim: number of tiles of dims (in any order)
        # :param tilings: number of tilings e.g. n=8, should be a power of 2 and >= 4k
        #  (k = number of included dimensions)
        # if unspecified or None, will use the lowest power of 2 above 4k
        # :param offset_per_dimension_fn: fn that takes the number of dimensions and returns offset units per dimension

        # Unpack parameters
        self._dims: Dims = dims
        included_dims: set[DimEnum] = tiling_group_parameters.included_dims
        tile_size_per_dim: Optional[dict[DimEnum, float]] = tiling_group_parameters.tile_size_per_dim
        tiles_per_dim: Optional[dict[DimEnum, int]] = tiling_group_parameters.tiles_per_dim
        tilings: Optional[int] = tiling_group_parameters.tilings
        offset_per_dimension_fn: Optional[Callable[[int], np.ndarray]] = tiling_group_parameters.offset_per_dimension_fn

        # convert included_dims to boolean arrays for each type of dimension, in correct order
        self._included_floats: np.ndarray = np.array(
            [dim in included_dims for dim in self._dims.state_float], dtype=bool)
        self._included_state_category: np.ndarray = np.array(
            [dim in included_dims for dim in self._dims.state_category], dtype=bool)
        self._included_action_category: np.ndarray = np.array(
            [dim in included_dims for dim in self._dims.action_category], dtype=bool)

        if tiles_per_dim is None:
            tiles_per_dim = dict()  # mutable

        # set a default for offset_per_dimension_fn if one is not passed
        if offset_per_dimension_fn is None:
            offset_per_dimension_fn = self._first_odd_integers

        # restrict remaining code to just the dimensions the tiling is over

        # included float dimensions in the correct order
        state_float_dims: dict[DimEnum, FloatDimension] = \
            {dim: float_dimension for dim, float_dimension in self._dims.state_float.items() if dim in included_dims}
        # float_dimensions = k in book
        float_dimension_count: int = len(state_float_dims)
        # range minimum in each dimension
        self._float_dim_min = np.array([float_dimension.min for float_dimension in state_float_dims.values()])

        # decided on the number of tilings to use and check value rules if passed in
        if tilings is None:
            if float_dimension_count == 0:
                self._tilings = 1   # category dimensions only
            else:
                self._tilings = 2 ** math.ceil(math.log2(4 * float_dimension_count))
        else:
            self._tilings: int = tilings
            if not math.log2(self._tilings).is_integer():
                # TODO: Replace print statements with logging
                print("Warning: tilings not a power of 2")
            if self._tilings < 4 * float_dimension_count:
                print("Warning: tilings < 4 * number of included float dimensions")

        # tiles needed per dimension ignoring offsets and wrapping for the moment
        # e.g. array([8, 8])

        # convert tile_size_per_dim entries so can just use tiles_per_dim
        if tile_size_per_dim:
            for dim, tile_size in tile_size_per_dim.items():
                if dim not in tiles_per_dim:
                    range_: float = self._dims.state_float[dim].range
                    tiles_per_dim[dim] = self.calc_tiles(range_, tile_size)
        # put in correct order and use a default if neccesary
        requested_tiles: list[int] = \
            [tiles_per_dim.get(dim, self._tilings) for dim, float_dimension in state_float_dims.items()]

        # calibrate input, dividing by the range and multiplying by the number of tiles. So a tile maps to 1.0.
        # e.g. array([8, 8]) / array([6.28318531, 6.28318531]) = array([1.27323954, 1.27323954])
        self._float_norm_tiles = np.array([
            float(tiles) / float_dimension.range
            for tiles, float_dimension in zip(requested_tiles, state_float_dims.values())
        ])

        # wrapped dimensions boolean array
        self._float_wrapped_dim = np.array([dr.wrap_around for dr in state_float_dims.values()], dtype=np.bool)
        self._has_wrapped: bool = np.sum(self._float_wrapped_dim) > 0

        # actual tiles per dimension (+ 1 extra tile needed for offset providing a full partitioning if not wrapped)
        # e.g. array([9, 9])
        self._actual_tiles = np.array([
            tiles + int(not float_dimension.wrap_around)
            for tiles, float_dimension in zip(requested_tiles, state_float_dims.values())
        ], dtype=np.int)
        self._actual_tiles_for_wrapped: np.ndarray = self._actual_tiles[self._float_wrapped_dim]

        # Calculate offsets

        offset_units_per_dimension = offset_per_dimension_fn(float_dimension_count)
        # e.g. array([1, 3])

        tilings_array = np.arange(self._tilings)
        # e.g. array([0, 1, 2, 3, 4, 5, 6, 7])

        tilings_spread = np.repeat([tilings_array], float_dimension_count, axis=0).T / float(self._tilings)
        """
        tilings spread evenly [0, 1) e.g. for 2 dimensions
        array([[0.   , 0.   ],
               [0.125, 0.125],
               [0.25 , 0.25 ],
               [0.375, 0.375],
               [0.5  , 0.5  ],
               [0.625, 0.625],
               [0.75 , 0.75 ],
               [0.875, 0.875]])
        """

        offset_lines = tilings_spread * offset_units_per_dimension  # element-wise multiplication
        """
        offset_lines is spread from [0, 1) and [0, 3) e.g.
        array([[0.   , 0.   ],
               [0.125, 0.375],
               [0.25 , 0.75 ],
               [0.375, 1.125],
               [0.5  , 1.5  ],
               [0.625, 1.875],
               [0.75 , 2.25 ],
               [0.875, 2.625]])              
        """

        self._float_offsets = offset_lines % 1
        """
        offset is mod 1 so spread 0 to 1 creating knight-move fill-in of space e.g.
        [[0.    0.   ]
         [0.125 0.375]
         [0.25  0.75 ]
         [0.375 0.125]
         [0.5   0.5  ]
         [0.625 0.875]
         [0.75  0.25 ]
         [0.875 0.625]]
        """

        self._state_category_possible_values = np.array([
            category_dimension.possible_values
            for dim, category_dimension in self._dims.state_category.items()
            if dim in included_dims
        ], dtype=np.int)

        self._action_category_possible_values = np.array([
            category_dimension.possible_values
            for dim, category_dimension in self._dims.action_category.items()
            if dim in included_dims
        ], dtype=np.int)

        self._total_tiles = self._tilings * \
            np.prod(self._actual_tiles) * \
            np.prod(self._state_category_possible_values) * \
            np.prod(self._action_category_possible_values)

        # tiling_dims e.g. array([9, 9])
        # 81 * array([0, 1, 2, 3, 4, 5, 6, 7])
        # = array([  0,  81, 162, 243, 324, 405, 486, 567])
        # where the index for each tiling starts from e.g. first tile of 2nd tiling is index 81
        # self._tile_base_index = self._tile_coder_base_index + np.prod(self._tiles_per_dim) * np.arange(self._tilings)

        # e.g. self._tiles_per_dim = array([9, 9]) then array([1, 9])
        # docs: "The product of an empty array is the neutral element 1"
        # how much to increase index for off_coords in each axis (e.g. +1 for x-axis, +9 for y-axis)
        # potentially a bit odd as it means the rows are worth 1 and the columns are worth 9
        # might have expected the rows to be worth 9 and then columns to be worth 1 but both work
        # self._coords_flattening_vector = np.array([np.prod(self._tiles_per_dim[0:i]) for i in range(dimensions)])

        # total number of tiles e.g. 648
        # self._total_tiles: int = self._tilings * np.prod(self._tiles_per_dim)

    @staticmethod
    def calc_tile_size(range_: float, tiles: int) -> float:
        return range_ / tiles

    @staticmethod
    def calc_tiles(range_: float, tile_size: float) -> int:
        if tile_size == 0.0:
            return 0
        else:
            return math.ceil(range_ / tile_size)

    def get_float_tile_coords(self, state_floats: np.ndarray) -> np.ndarray:
        """
        :param state_floats: a full set of floats for the state
        :return: array of: tilings x included dimensions
        """
        float_tile_coords = np.floor(
            # state normalised so states from zero and tile width is 1.0
            (state_floats[self._included_floats] - self._float_dim_min) * self._float_norm_tiles
            + self._float_offsets                               # broadcast to all tilings with their offsets
        ).astype(np.int)
        if self._has_wrapped:
            float_tile_coords[:, self._float_wrapped_dim] %= self._actual_tiles_for_wrapped
        return float_tile_coords

    def filter_state_categories(self, state_categories: np.ndarray) -> np.ndarray:
        return state_categories[self._included_state_category]

    def filter_action_categories(self, action_categories: np.ndarray) -> np.ndarray:
        return action_categories[self._included_action_category]

    def get_all_tile_coords(self) -> list[tuple]:
        range_sizes: list[int] = [self._tilings]
        range_sizes.extend([int(tiles) for tiles in self._actual_tiles])
        range_sizes.extend([int(pv) for pv in self._state_category_possible_values])
        range_sizes.extend([int(pv) for pv in self._action_category_possible_values])

        ranges: list = []
        for range_size in range_sizes:
            ranges.append(range(range_size))

        tile_coord_list: list[tuple] = list(itertools.product(*ranges))
        return tile_coord_list

        # for tile_count in self._actual_tiles.shape:
        #     range_sizes.append(tile_count)
        # for value_count in self._state_category_possible_values.shape
        #
        # # tilings_array = np.arange(self._tilings)
        # # # e.g. array([0, 1, 2, 3, 4, 5, 6, 7])
        # tile_coords: list[int]
        #
        # for tiling in range(self._tilings):
        #     tile_coords = [tiling]
        #     for tile_count in self._actual_tiles.shape:
        #         for tile_coord in range(tile_count):
        #             tile_coords.append(tile_coord)
        #
        #
        # self._total_tiles = self._tilings * \
        #     np.prod(self._actual_tiles) * \
        #     np.prod(self._state_category_possible_values) * \
        #     np.prod(self._action_category_possible_values)

    @staticmethod
    def _first_odd_integers(number_of_dimensions: int) -> np.ndarray:  # e.g. 2 -> array([1, 3])
        return 2 * np.arange(number_of_dimensions) + 1

    @property
    def total_tiles(self) -> int:
        return self._total_tiles

    @property
    def tilings(self) -> int:
        return self._tilings
