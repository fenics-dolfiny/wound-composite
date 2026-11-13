import numpy as np

from wound_composite.layout import Layout

__all__ = [
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
