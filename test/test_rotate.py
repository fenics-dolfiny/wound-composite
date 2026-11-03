import basix.ufl
import numpy as np
import pytest
import ufl
from numpy import typing as npt

from wound_composite.la import rotate

random_gen = np.random.default_rng(0)


@pytest.mark.parametrize("component", list(range(3)))
@pytest.mark.parametrize("angle", np.linspace(0, np.pi, 5, endpoint=True))
def test_principal(component: int, angle: float) -> None:
    axis = np.zeros(3)
    axis[component] = 1
    axis = axis
    axis = ufl.as_vector(axis)

    domain = ufl.Mesh(basix.ufl.element("P", "tetrahedron", degree=1, shape=(3,)))
    x = ufl.SpatialCoordinate(domain)
    expr = rotate(x, axis, angle)

    # components of rotation plane
    x_comp = (component + 1) % 3
    y_comp = (component + 2) % 3

    p = (1, 2, 3)
    assert np.isclose(expr[component](p), p[component])
    assert np.isclose(expr[x_comp](p), p[x_comp] * np.cos(angle) - p[y_comp] * np.sin(angle))
    assert np.isclose(expr[y_comp](p), p[x_comp] * np.sin(angle) + p[y_comp] * np.cos(angle))


@pytest.mark.parametrize("vec", random_gen.random((5, 3)))
@pytest.mark.parametrize("axis", random_gen.random((5, 3)))
@pytest.mark.parametrize("angle", np.linspace(0, np.pi, 5, endpoint=True))
def test_unit_length(vec: npt.NDArray, axis: npt.NDArray, angle: float) -> None:
    axis /= np.linalg.norm(axis)

    domain = ufl.Mesh(basix.ufl.element("P", "tetrahedron", degree=1, shape=(3,)))
    x = ufl.SpatialCoordinate(domain)
    expr = rotate(x, ufl.as_vector(axis), angle)

    val = np.array([expr[0](tuple(vec)), expr[1](tuple(vec)), expr[2](tuple(vec))])
    assert np.isclose(np.linalg.norm(val), np.linalg.norm(vec))


def test_multi_axes() -> None:
    """Test (consecutive) rotation around multiple axes."""
    domain = ufl.Mesh(basix.ufl.element("P", "tetrahedron", degree=1, shape=(3,)))
    x = ufl.SpatialCoordinate(domain)

    axis_0 = np.array([1, 1, 0], dtype=np.float64)
    axis_0 /= np.linalg.norm(axis_0)
    angle_0 = np.pi
    axis_1 = np.array([0, 1, 0], dtype=np.float64)
    angle_1 = np.pi

    # 1. rotate around ax-0 then around rotated! ax-1
    expr = rotate(rotate(x, ufl.as_vector(axis_1), angle_1), ufl.as_vector(axis_0), angle_0)

    p = (1, 2, 3)
    val = np.array([expr[0](p), expr[1](p), expr[2](p)])
    assert np.allclose(val, [2, -1, 3])

    # does not respect the rotation in the second rotational axis
    expr = rotate(rotate(x, ufl.as_vector(axis_0), angle_0), ufl.as_vector(axis_1), angle_1)
    val = np.array([expr[0](p), expr[1](p), expr[2](p)])
    assert not np.allclose(val, [2, -1, 3])
