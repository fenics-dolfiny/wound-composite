from pathlib import Path

import dolfinx
import dolfiny
import numpy as np
import pyvista as pv
import ufl
import ufl.classes
from dolfiny.expression import norm, normalize
from mpi4py import MPI
from petsc4py import PETSc

from wound_composite.la import rotate, tangents
from wound_composite.layout.templates import regular_grid

output_dir = Path(__file__).parent / Path(__file__).stem
output_dir.mkdir(exist_ok=True)

comm = MPI.COMM_WORLD
if comm.size > 1:
    raise RuntimeError("Parallelization not supported.")

# Create regularly spaced grid bushing setup
layout = regular_grid(n=(n := np.array([20, 20])), spacing=(spacing := np.array([1.0, 1.0])))
print(f"no. of strings: {layout.cells.shape[0]}")

mesh = layout.mesh


# Define function spaces
#
# Offset from "ideal" straight connection between bushing centres
O0 = dolfinx.fem.functionspace(mesh, ("DG", 1, (3,)))

# Displacement and rotational dofs
V_u = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
V_r = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))

# Define functions
o0 = layout.offset

u = dolfinx.fem.Function(V_u, name="displacement")
r = dolfinx.fem.Function(V_r, name="rotation")

orientation = layout.orientation

# Define boundary conditions
mesh.topology.create_connectivity(0, 1)

vertices_fixed = dolfinx.mesh.locate_entities(
    mesh,
    0,
    lambda x: (
        np.isclose(x[0], 0)
        | np.isclose(x[0], (n[0] - 1) * spacing[0])
        | np.isclose(x[1], 0)
        | np.isclose(x[1], (n[1] - 1) * spacing[1])
    ),
)
dofs_u_fixed = dolfinx.fem.locate_dofs_topological(V_u, 0, vertices_fixed)

bcs = [dolfinx.fem.dirichletbc(np.array([0, 0, 0], dtype=np.float64), dofs_u_fixed, V_u)]

# Structure: load parameters

F_ext = dolfinx.fem.Function(V_u)
x = ufl.SpatialCoordinate(mesh)
min_x = np.min(mesh.geometry.x[:, 0])
max_x = np.max(mesh.geometry.x[:, 0])
assert np.allclose(min_x, 0)
min_y = np.min(mesh.geometry.x[:, 1])
max_y = np.max(mesh.geometry.x[:, 1])
assert np.allclose(min_y, 0)

F_ext.interpolate(
    dolfinx.fem.Expression(
        ufl.as_vector(
            (0, 0, ufl.sin(x[0] * 2 * np.pi / max_x) * ufl.sin(x[1] * 2 * np.pi / max_y))
        ),
        V_u.element.interpolation_points,
    )
)
F_ext.x.array[:] *= 10000

vertices_load = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.full_like(x[0], True))
dofs_load = dolfinx.fem.locate_dofs_topological(V_u, 0, vertices_load)

# Structure: cross section geometry (circular)
radius = 2.5 * 1e-4  # [m]
A = np.pi * radius**2  # [m^2]

# Structure: material parameters
nu = 0.0  # [-] Poisson ratio


def E(ε: ufl.classes.Expr) -> ufl.classes.Expr:
    # Young's modulus
    E_tension = 100 * 1e9  # [N/m^2] elasticity modulus (tension)
    E_compression = 0.5 * E_tension  # [N/m^2] elasticity modulus (compression)

    ε_tangential = ufl.inner(ε * gs, gs)
    return ufl.conditional(ε_tangential >= 0, E_tension, E_compression)  # type: ignore


def lamé_λ(ε: ufl.classes.Expr) -> ufl.classes.Expr:
    # Lamé constant λ
    return E(ε) * nu / (1 + nu) / (1 - 2 * nu)  # type: ignore


def lamé_μ(ε: ufl.classes.Expr) -> ufl.classes.Expr:
    # Lamé constant μ
    return E(ε) / 2 / (1 + nu)  # type: ignore


def S(ε: ufl.classes.Expr) -> ufl.classes.Expr:
    """Stress as function of strain from strain energy function."""
    ε = ufl.variable(ε)
    W = lamé_μ(ε) * ufl.inner(ε, ε) + lamé_λ(ε) / 2 * ufl.tr(ε) ** 2  # Saint-Venant Kirchhoff
    S = ufl.diff(W, ε)
    return S  # type: ignore


δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V_u, V_r))

# Define state as (ordered) list of functions
m = [u, r]

# Undeformed configuration
x0 = ufl.SpatialCoordinate(mesh)

x1 = x0 + o0

t0, t1 = tangents(orientation)
rotated_o0 = rotate(rotate(rotate(o0, orientation, r[2]), t1, r[1]), t0, r[0])
o_rot = ufl.interpolate(rotated_o0, O0)

x2 = x0 + o_rot
x3 = x2 + u

# Deformation gradient between undeformed and reference configuration
J_01 = ufl.grad(x1)


def grad_ref(f: ufl.classes.Expr) -> ufl.classes.Expr:
    """Gradient wrt. reference x_{-1}."""
    # workaround for ufl.ref_grad() broken:
    # d/dx_{-1} = d/dx_0 dx_0/dx_{-1}
    return ufl.grad(f) * ufl.geometry.Jacobian(mesh)  # type: ignore


def grad_1(f: ufl.classes.Expr) -> ufl.classes.Expr:
    """Gradient wrt. undeformed x_1."""
    # d/dx_1 = d/dx0 dx0/dx1 = grad( . ) * inv(grad(x1))
    return ufl.grad(f) * ufl.inv(J_01)  # type: ignore


# Deformation gradient between deformed and undeformed configuration
J_13 = grad_1(x3)

# Deformation gradient between undeformed and undeformed configuration
J_11 = ufl.Identity(3)

# Green-Lagrange strains (total): determined by deformation kinematics
B = (J_13.T * J_13 - J_11.T * J_11) / 2

# Unit tangent basis in undeformed configuration
gs = normalize(grad_ref(x1)[:, 0])

P = ufl.outer(gs, gs)

# Membrane strain
Em = P * B * P

# Stress resultant tensors
N = S(Em) * A  # [N] = kg/(m s^2) m^2 = N
N_tangential = ufl.inner(N * gs, gs)

# Weak form: components (as one-form)
meshtags = dolfinx.mesh.meshtags(mesh, 0, vertices_load, 1)
dP = ufl.Measure("dP", domain=mesh, subdomain_data=meshtags)

# dx = ufl.dx # over estimates drastically, 30 is necessary, otherwise
#               iteration count explodes or divergence.
dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 30})

gs0 = normalize(grad_ref(x0)[:, 0])


def W_density(ε: ufl.classes.Expr) -> ufl.classes.Expr:
    """Saint-Venant Kirchhoff."""
    W = lamé_μ(ε) * ufl.inner(ε, ε) + lamé_λ(ε) / 2 * ufl.tr(ε) ** 2
    return W  # type: ignore


Pi_mech = W_density(Em) * A * norm(J_01 * gs0) * dx
Pi_ext = ufl.inner(u, F_ext) * dP(1)

Pi = Pi_mech - Pi_ext

residual = [ufl.derivative(Pi, _u, _δu) for _u, _δu in zip(m, δm)]

# Options for PETSc backend
opts = PETSc.Options(name := "solver")

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "l2"
opts["snes_rtol"] = 1e-12
opts["snes_max_it"] = 100
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(residual, m, prefix=name, bcs=bcs)

# Solve nonlinear problem
u.x.array[:] = 0.0
r.x.array[:] = 0.0

problem.solve(u_init=[u, r])
problem.status(error_on_failure=True)


# Post process
def interpolate_O0(expr: ufl.classes.Expr, name: str = "") -> dolfinx.fem.Function:
    f = dolfinx.fem.Function(O0, name=name)
    expr_compiled = dolfinx.fem.Expression(expr, O0.element.interpolation_points)
    f.interpolate(expr_compiled)
    return f


# TODO: interpolate expression support
int_rotated_o0 = interpolate_O0(rotated_o0, "rotated_o0")

V_N = dolfinx.fem.functionspace(mesh, ("DG", 0))
N_f = dolfinx.fem.Function(V_N, name="normal force")
N_f.interpolate(
    dolfinx.fem.Expression(
        ufl.replace(N_tangential, {o_rot: int_rotated_o0}), V_N.element.interpolation_points
    )
)

V_N_o = dolfinx.fem.functionspace(mesh, ("DG", 1))
N_f_o = dolfinx.fem.Function(V_N_o, name="normal force")
N_f_o.interpolate(N_f)


with dolfinx.io.VTXWriter(
    comm,
    f"{output_dir / Path(__file__).stem}_output.bp",
    [
        o0,
        int_u := interpolate_O0(u, u.name),
        int_rotated_o0,
        interpolate_O0(t0, "tangent 0"),
        interpolate_O0(t1, "tangent 1"),
        N_f_o,
    ],
) as writer:
    writer.write(0.0)

# Scripted plotting
vtk_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(O0))
vtk_grid.point_data[o0.name] = o0.x.array.reshape(-1, O0.dofmap.index_map_bs)
vtk_grid.point_data[N_f_o.name] = N_f_o.x.array
vtk_grid.point_data["displaced"] = (int_rotated_o0.x.array + int_u.x.array).reshape(
    -1, O0.dofmap.index_map_bs
)

pl = pv.Plotter(
    window_size=[res := 2048 * 2, res // 2],
    shape=(1, 2),
    off_screen=True,
    theme=dolfiny.pyvista.theme,
)

mesh_style = {
    "line_width": 2,
    "render_lines_as_tubes": True,
    "color": "black",
}

point_style = {
    "color": "red",
    "render_points_as_spheres": True,
    "point_size": 6 // 2,
}

# undeformed - xy
pl.subplot(0, 0)
pl.parallel_projection = True
vtk_grid_offset = vtk_grid.warp_by_vector(o0.name)
vtk_grid_offset.set_active_scalars(None)
pl.add_mesh(vtk_grid_offset, **mesh_style)  # type: ignore
pl.add_points(vtk_grid_offset.points, **point_style)  # type: ignore
pl.show_axes()  # type: ignore

pl.subplot(0, 1)
pl.parallel_projection = True
vtk_grid_deformed = vtk_grid.warp_by_vector("displaced")
pl.add_mesh(
    vtk_grid_deformed,
    scalars=N_f_o.name,
    n_colors=10,
    **mesh_style,  # type: ignore
)
pl.add_points(vtk_grid_deformed.points, **point_style)  # type: ignore
pl.show_axes()  # type: ignore

pl.save_graphic(output_dir / "plot.pdf")
