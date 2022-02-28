import numpy as np
import matplotlib.pyplot as plt
import time
import enum
from dataclasses import dataclass

from mdp.model.feature.tile_coding.tile_coding import TileCoding
from mdp.model.environment.non_tabular.dimension.float_dimension import FloatDimension
from mdp.model.environment.non_tabular.dimension.category_dimension import CategoryDimension
from mdp.model.environment.non_tabular.dimension.dim_enum import DimEnum
from mdp.model.environment.non_tabular.dimension.dims import Dims
from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState


class Dim(DimEnum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


@dataclass(frozen=True)
class State(NonTabularState):
    x: float
    y: float
    z: int

    def _get_floats(self) -> list[float]:
        return [self.x, self.y]

    def _get_categories(self) -> list:
        return [self.z]


def main():
    # tile coder dimensions, limits, tilings
    dims: Dims = Dims()
    dims.state_float[Dim.X] = FloatDimension(min=0.0, max=2.0 * np.pi, wrap_around=False)
    dims.state_float[Dim.Y] = FloatDimension(min=0.0, max=2.0 * np.pi, wrap_around=False)
    dims.state_category[Dim.Z] = CategoryDimension(possible_values=6)

    # convinence only
    dim_x = dims.state_float[Dim.X]
    dim_y = dims.state_float[Dim.Y]
    # dim_z = dims.state_category[Dim.Z]

    tile_coding = TileCoding(dims=dims)
    # tile_coding = TileCoding(dimension_ranges=dimensions_ranges, max_size=800, use_dict=False)

    # tile_size_per_dim = [dr for dr, tiles in dimensions_ranges, [8,8,0] ]
    # tile_coding.add(tile_size_per_dim=np.array([2.0 * np.pi/8, 2.0 * np.pi/8, 0]), tilings=8)

    # tile_coding.add(included_dims=np.array([True, False, False]), tilings=2 ** 4)  # , tilings=2 ** 4
    # tile_coding.add(included_dims=np.array([False, True, False]), tilings=2 ** 4)  # , tilings=2 ** 4
    # tile_coding.add(included_dims=np.array([False, False, True]), tilings=2 ** 4)  # , tilings=2 ** 4

    tile_coding.add(included_dims={Dim.X, Dim.Y})  # , tilings=2 ** 4
    tile_coding.add(included_dims={Dim.Y, Dim.Z})  # , tilings=2 ** 4
    tile_coding.add(included_dims={Dim.X, Dim.Z})  # , tilings=2 ** 4

    tilings = tile_coding.tilings
    print(f"total tilings = {tile_coding.tilings}")
    print(f"max_size = {tile_coding.max_size}")

    # linear function weight vector, step size for SGD
    w = np.zeros(tile_coding.max_size)
    alpha = 0.1 / tilings

    # take 10,000 samples of target function, output mse of batches of 100 points
    timer = time.time()
    batch_size = 100
    for batches in range(100):
        mse = 0.0
        for b in range(batch_size):
            x = dim_x.min + np.random.rand() * dim_x.range
            y = dim_y.min + np.random.rand() * dim_y.range
            z = np.random.randint(5, 10 + 1)
            target = target_ftn(x, y, z)
            # state = State(is_terminal=False, x=x, y=y, z=z)
            tile_coding.state = State(is_terminal=False, x=x, y=y, z=z)
            vector = tile_coding.vector
            w[vector] += alpha * (target - w[vector].sum())
            mse += (target - w[vector].sum()) ** 2
        mse /= batch_size
        # print('samples:', (batches + 1) * batch_size, 'batch_mse:', mse)
    print('elapsed time:', time.time() - timer)

    # get learned function
    # print('mapping function...')
    resolution = 200
    x = np.arange(dim_x.min, dim_x.max, dim_x.range / resolution)
    y = np.arange(dim_y.min, dim_y.max, dim_y.range / resolution)
    z = np.zeros([len(x), len(y)])
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            tile_coding.state = State(is_terminal=False, x=xi, y=yj, z=7)
            z[i, j] = tile_coding.dot_product_full_vector(w)

    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    mesh_x, mesh_y = np.meshgrid(x, y)
    ax.plot_surface(mesh_x, mesh_y, z, cmap=plt.get_cmap('hot'))
    fig.tight_layout(pad=0)
    plt.show()


# target function with gaussian noise
def target_ftn(x: float, y: float, z: int) -> float:
    # return x/(2.0*np.pi) + np.cos(y) + 0.1 * np.random.randn()
    return np.sin(x) * np.cos(y) * z/10 + 0.1 * np.random.randn()


if __name__ == '__main__':
    main()
