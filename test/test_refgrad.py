import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

"""
Checks workaround for

    df/dX = dx/dX^T df/dx

i.e.

    ulf.classes.ReferenceGrad(f) = ufl.Jacobian(mesh).T * ufl.grad(f)
"""


def test_1d() -> None:
    # x = 1/2 * X iff. X = 2x
    mesh = dolfinx.mesh.create_interval(MPI.COMM_SELF, 1, [0, 0.5])

    V = dolfinx.fem.functionspace(mesh, ("P", 1))

    # f(a) = a
    f = dolfinx.fem.Function(V)
    f.interpolate(
        dolfinx.fem.Expression(ufl.SpatialCoordinate(mesh), V.element.interpolation_points)
    )

    # form = dolfinx.fem.form(ufl.classes.ReferenceGrad(f)[0] * ufl.dx)
    form = dolfinx.fem.form((ufl.Jacobian(mesh).T * ufl.grad(f))[0] * ufl.dx)
    val = dolfinx.fem.assemble_scalar(form)

    # f(X) = f(1/2x) = 1/2*X
    # refgrad f(x) = grad f(X) = 1/2
    # int from 0 to .5 of grad(f(X)) = .5^2
    assert np.allclose(val, 0.25)
