from functools import cached_property

import basix
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from numpy import typing as npt


class Layout:
    _x: npt.NDArray[np.float64]
    _cells: npt.NDArray[np.int64]
    _orientation: npt.NDArray[np.float64]
    _offset_h: npt.NDArray[np.float64]
    _offset_r: npt.NDArray[np.float64]

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        cells: npt.NDArray[np.int64],
        orientation: npt.NDArray[np.float64],
        offset_h: npt.NDArray[np.float64],
        offset_r: npt.NDArray[np.float64],
    ):
        assert len(x.shape) == 2
        assert x.shape[1] == 3

        assert len(cells.shape) == 2
        assert cells.shape[1] == 2
        # assert np.all(cells[:, 0] <= cells[:, 1])

        assert len(orientation.shape) == 2
        assert orientation.shape[1] == 3
        assert np.allclose(np.linalg.norm(orientation, axis=1), 1)

        assert len(offset_h.shape) == 2
        assert offset_h.shape[1] == 2

        assert len(offset_r.shape) == 2
        assert offset_r.shape[1] == 2

        # TODO: sizes between data arrays

        self._x = x
        self._cells = cells
        self._orientation = orientation
        self._offset_h = offset_h
        self._offset_r = offset_r

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """Array of bushing centers, has shape (-1, 3)."""
        return self._x

    @property
    def cells(self) -> npt.NDArray[np.int64]:
        """Array of cells, has shape (-1, 2).

        Note:
            The stored indices map into x.

        """
        return self._cells

    @property
    def orientation_array(self) -> npt.NDArray[np.float64]:
        """Array of orientations of bushings, has shape (-1, 3).

        Note:
            x.shape[0] == orientation.shape[0]

        """
        return self._orientation

    @property
    def offset_h(self) -> npt.NDArray[np.float64]:
        """Offset start/end of string along orientation axis.

        Returns:
            shape (-1, 2).

        """
        return self._offset_h

    @property
    def offset_r(self) -> npt.NDArray[np.float64]:
        """Offset start/end of string along normal direction.

        Note:
            The normal direction for a fibre going from a to b, with
            bushing orientations o_a, o_b has normal directions
                        (b - a) x o_a
                n_a = -----------------
                      ||(b - a) x o_a||

                        (b - a) x o_b
                n_b = -----------------
                      ||(b - a) x o_b||

            The relative direction of the edge is not changed, therefore
            same sign of offsets implies ending on same side.

        Returns:
            shape (-1, 2)

        """
        return self._offset_r

    @cached_property
    def mesh(self) -> dolfinx.mesh.Mesh:
        return create_mesh(self)

    @cached_property
    def orientation(self) -> dolfinx.fem.Function[np.float64]:
        """Orientation of bushings.

        Note: CG 1 function - only ever evaluate at vertices!

        """
        V = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 1, (3,)))
        f = dolfinx.fem.Function(V, name="orientation", dtype=np.float64)
        assign_orientation(self, f)
        return f  # type: ignore

    @cached_property
    def offset(self) -> dolfinx.fem.Function[np.float64]:
        """Offset start/end of string along orientation axis.

        Note: DG 1 function - only ever evaluate at vertices!

        """
        V = dolfinx.fem.functionspace(self.mesh, ("DG", 1, (3,)))
        f = dolfinx.fem.Function(V, name="offset", dtype=np.float64)
        assign_offset(self, f)
        return f  # type: ignore


def assign_orientation(layout: Layout, orientation: dolfinx.fem.Function) -> None:
    """Assign the bushing orientations to a CG1^3 function."""
    basix_element = orientation.function_space.element.basix_element

    if basix_element.family != basix.ElementFamily.P:
        raise RuntimeError("Orientation only supported on Lagrange space.")

    if basix_element.discontinuous:
        raise RuntimeError("Orientation is a continuous quantity.")

    if basix_element.degree != 1:
        raise RuntimeError("Only element wise affine supported.")

    if not np.allclose(basix_element.value_shape, [3]):
        raise RuntimeError("Offset is a quantity with shape (3,).")

    V = orientation.function_space
    bs = V.dofmap.bs

    for vertex in range(layout.x.shape[0]):
        # TODO: proper dof mapping?
        orientation.x.array[vertex * bs : (vertex + 1) * bs] = layout.orientation_array[vertex]


def assign_offset_h(layout: Layout, u: dolfinx.fem.Function) -> None:
    """Assign to function u the offset in axis aligned direction."""
    basix_element = u.function_space.element.basix_element

    if basix_element.family != basix.ElementFamily.P:
        raise RuntimeError("Offset only supported on Lagrange space.")

    if not basix_element.discontinuous:
        raise RuntimeError("Offset is a discontinuous quantity.")

    if basix_element.degree != 1:
        raise RuntimeError("Only element wise affine supported.")

    if not np.allclose(basix_element.value_shape, [3]):
        raise RuntimeError("Offset is a quantity with shape (3,).")

    bs = u.function_space.dofmap.bs
    dofmap = u.function_space.dofmap.list

    for cell, (vertex_a, vertex_b) in enumerate(layout.cells):
        orient_a = layout.orientation_array[vertex_a]
        orient_b = layout.orientation_array[vertex_b]

        a = layout.offset_h[cell, 0] * orient_a

        dof_a = dofmap[cell][0]
        u.x.array[dof_a * bs : (dof_a + 1) * bs] = a

        b = layout.offset_h[cell, 1] * orient_b

        dof_b = dofmap[cell][1]
        u.x.array[dof_b * bs : (dof_b + 1) * bs] = b


def assign_offset(layout: Layout, u: dolfinx.fem.Function) -> None:
    """Assign to function u the offset of the layout.

    The undeformed configuration of the GRAM structure is then given by
    x + u, where x is the reference coordinate system.
    """
    basix_element = u.function_space.element.basix_element

    if basix_element.family != basix.ElementFamily.P:
        raise RuntimeError("Offset only supported on Lagrange space.")

    if not basix_element.discontinuous:
        raise RuntimeError("Offset is a discontinuous quantity.")

    if basix_element.degree != 1:
        raise RuntimeError("Only element wise affine supported.")

    if not np.allclose(basix_element.value_shape, [3]):
        raise RuntimeError("Offset is a quantity with shape (3,).")

    bs = u.function_space.dofmap.bs
    dofmap = u.function_space.dofmap.list

    assign_offset_h(layout, u)

    # TODO: if need -> move to assign_offset_r
    for cell, (vertex_a, vertex_b) in enumerate(layout.cells):
        orient_a = layout.orientation_array[vertex_a]
        orient_b = layout.orientation_array[vertex_b]

        a_to_b = layout.x[vertex_b] - layout.x[vertex_a]

        n_a_to_b_orient_a = np.cross(a_to_b, orient_a)
        n_a_to_b_orient_a /= np.linalg.norm(n_a_to_b_orient_a)

        a = layout.offset_r[cell, 0] * n_a_to_b_orient_a

        dof_a = dofmap[cell][0]
        u.x.array[dof_a * bs : (dof_a + 1) * bs] += a

        n_a_to_b_orient_b = np.cross(a_to_b, orient_b)
        n_a_to_b_orient_b /= np.linalg.norm(n_a_to_b_orient_b)

        b = layout.offset_r[cell, 1] * n_a_to_b_orient_b

        dof_b = dofmap[cell][1]
        u.x.array[dof_b * bs : (dof_b + 1) * bs] += b


def create_mesh(layout: Layout) -> dolfinx.mesh.Mesh:
    """Create computational domain/mesh from layout."""
    comm = MPI.COMM_SELF

    # 1) topology
    cell_types = [dolfinx.mesh.CellType.interval]
    cells = layout.cells.ravel()
    orig_index = np.arange(cells.size, dtype=np.int64)
    ghost_owner = np.array([], dtype=np.int32)
    boundary_vertices = np.array([], dtype=np.int64)

    topology = dolfinx.cpp.mesh.create_topology(
        comm,
        cell_types,
        [cells],
        [orig_index],
        [ghost_owner],
        boundary_vertices,
        1,
    )

    # 2) geometry
    coordinate_elements: list[dolfinx.cpp.fem.CoordinateElement_float64] = [
        dolfinx.fem.coordinate_element(cell_types[0], 1)._cpp_object  # type: ignore
    ]
    nodes = np.arange(layout.x.shape[0], dtype=np.int64)
    xdofs = layout.cells.flatten()  # linear geometry assumed
    x = layout.x.flatten()
    gdim = 3

    geom = dolfinx.cpp.mesh.create_geometry(topology, coordinate_elements, nodes, xdofs, x, gdim)

    # 3) mesh
    cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(comm, topology, geom)
    ufl_mesh = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(3,)))
    return dolfinx.mesh.Mesh(cpp_mesh, ufl_mesh)
