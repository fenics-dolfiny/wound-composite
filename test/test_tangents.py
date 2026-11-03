import numpy as np
import pytest
import ufl
from numpy import typing as npt

from wound_composite.la import tangents


@pytest.mark.parametrize(
    "normal_arr",
    np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        ]
    ),
)
@pytest.mark.parametrize("sign", 2 * ((np.arange(8)[:, None] >> np.arange(3)) & 1) - 1)
def test(normal_arr: npt.NDArray, sign: npt.NDArray) -> None:
    normal_arr *= sign

    normal = ufl.as_vector(normal_arr)

    t0, t1 = tangents(normal)

    p = (1, 2, 3)  # to trick UFL evaluate

    # orthogonal
    assert np.isclose(ufl.inner(t0, normal)(p), 0)
    assert np.isclose(ufl.inner(t1, normal)(p), 0)
    assert np.isclose(ufl.inner(t0, t1)(p), 0)

    # unit length
    assert np.isclose(ufl.inner(normal, normal)(p), 1)
    assert np.isclose(ufl.inner(t0, t0)(p), 1)
    assert np.isclose(ufl.inner(t1, t1)(p), 1)
