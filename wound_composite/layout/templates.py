import numpy as np
from numpy import typing as npt
from scipy.spatial import distance

from wound_composite.layout import Layout

__all__ = [
    "regular_grid",
    "triad",
]


def triad() -> Layout:
    x = np.asarray([[0, 0, 0], [3, 0, 0], [0, -1, -3]], dtype=np.float64)
    orientation = np.asarray(
        [[0, 1, 0], [0, 1, 0], [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]],
        dtype=np.float64,
    )

    cells = np.asarray([[0, 1], [0, 1], [0, 2], [0, 2], [0, 2], [1, 2], [1, 2]], dtype=np.int64)
    cells = np.repeat(cells, 2, axis=0)

    height = 0.5
    offset_h = np.asarray(
        [
            [height, -height],
            [-height, -height],
            [height, height],
            [height, -height],
            [-height, -height],
            [-height, -height],
            [-height, height],
        ],
        dtype=np.float64,
    )
    offset_h = np.repeat(offset_h, 2, axis=0)

    radius = 0.2
    offset_r = np.asarray(
        [
            [radius, radius],
            [-radius, -radius],
        ],
        dtype=np.float64,
    )
    offset_r = np.tile(offset_r, (offset_h.shape[0] // 2, 1))

    return Layout(x, cells, orientation, offset_h, offset_r)


def regular_grid(
    n: npt.NDArray[np.int32] = np.array([10, 10], dtype=np.int32),
    spacing: npt.NDArray[np.float64] = np.array([1, 1], dtype=np.float64),
    radius: float = 1.5,
) -> Layout:
    def centers() -> npt.NDArray[np.float64]:
        size = (n - 1) * spacing
        x = np.linspace(0, size[0], n[0])
        y = np.linspace(0, size[1], n[1])
        grid = np.meshgrid(x, y)
        coords: npt.NDArray[np.float64] = np.vstack([_x.flatten() for _x in grid]).transpose()
        coords = np.c_[coords, np.zeros(coords.shape[0])]
        return coords

    x = centers()
    orientation = np.tile(np.asarray([0, 0, 1], dtype=np.float64), x.shape[0]).reshape(-1, 3)

    dist = distance.cdist(x, x, "euclidean")
    indices = np.where((dist > 0) & (dist <= radius))
    cells = np.vstack(indices).transpose()
    cells = np.sort(cells, axis=1)
    cells = np.unique(cells, axis=0)
    cells = np.repeat(cells, 8, axis=0)

    height = 0.5
    offset_h = np.asarray(
        [
            [height, height],
            [height, height],
            [height, -height],
            [height, -height],
            [-height, height],
            [-height, height],
            [-height, -height],
            [-height, -height],
        ],
        dtype=np.float64,
    )
    offset_h = np.tile(offset_h, (cells.shape[0] // 8, 1))

    radius = 0.2
    offset_r = np.asarray([[radius, radius], [-radius, -radius]], dtype=np.float64)
    offset_r = np.tile(offset_r, (offset_h.shape[0] // 2, 1))

    return Layout(x, cells, orientation, offset_h, offset_r)
