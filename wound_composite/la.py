import ufl
from dolfiny.expression import normalize


def rotate(
    x: ufl.classes.Expr, axis: ufl.classes.Expr, angle: ufl.classes.Expr | float
) -> ufl.classes.Expr:
    """Apply rotation to a vector around axis by given angle.

    This implements the action of the rotation matrix.

    Args:
        x: vector to rotate
        axis: rotation axis
        angle: angle (in radians)

    Returns:
        Expression with applied rotation.

    Note:
        The axis vector needs to be of unit length.

    """
    assert x.ufl_shape == (3,)
    assert axis.ufl_shape == (3,)

    # Rodrigues formula
    R = x * ufl.cos(angle)
    R += ufl.cross(axis, x) * ufl.sin(angle)
    R += axis * ufl.inner(axis, x) * (1 - ufl.cos(angle))

    # Rodrigues formula in projector form
    # R = axis * ufl.inner(axis, x)  # axial
    # R += ufl.cos(angle) * ufl.cross(ufl.cross(axis, x), axis)
    # # perpendicular part (in plane)
    # R += ufl.sin(angle) * ufl.cross(axis, x)  # orthogonal part

    # https://doi.org/10.1119/1.10264 eq. 5.
    # TODO: fails on float bounds in unit tests - @MisterRounding?
    # R = x * ufl.cos(angle)
    # R += axis * ufl.inner(x, axis) * (1 - ufl.cos(angle))
    # R -= ufl.cross(x, axis) * ufl.sin(angle)

    return R  # type: ignore


def tangents(n: ufl.classes.Expr) -> tuple[ufl.classes.Expr, ufl.classes.Expr]:
    """Compute tangent vectors spanning tangent plane to normal.

    Args:
        n: normal vector

    Returns:
        tuple of two orthonormal vectors spanning the tangent plane.

    """
    # a arbitrary, not parallel to n
    a = ufl.conditional(abs(n[0]) < 0.5, ufl.as_vector((1, 0, 0)), ufl.as_vector((0, 1, 0)))

    #            a x n
    #    t0 =  --------
    #         || a x n ||
    t0 = normalize(ufl.cross(a, n))

    # t1 = n x t0
    t1 = ufl.cross(n, t0)

    return t0, t1
