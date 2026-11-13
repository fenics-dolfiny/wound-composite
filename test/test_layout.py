import dolfinx
import numpy as np
import ufl

from wound_composite.layout import Layout


def test_meshing(layout: Layout) -> None:
    mesh = layout.mesh

    cell_im = mesh.topology.index_map(1)
    assert cell_im.size_local == cell_im.size_global == layout.cells.shape[0]

    vertex_im = mesh.topology.index_map(0)
    assert vertex_im.size_local == vertex_im.size_global == layout.x.shape[0]

    c_to_v = mesh.topology.connectivity(1, 0)
    assert np.all(layout.cells.flatten() == c_to_v.array)

    vol = np.float64(0.0)
    for cell in layout.cells:
        a, b = layout.x[cell]
        vol += np.linalg.norm(a - b)

    assert dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(mesh))) == vol


def test_offset(layout: Layout) -> None:
    assert layout.offset


def test_orientation(layout: Layout) -> None:
    assert layout.orientation
