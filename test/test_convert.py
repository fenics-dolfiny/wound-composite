import gmsh

from wound_composite import convert
from wound_composite.layout import Layout


def test_to_gmsh(layout: Layout) -> None:
    gmsh.initialize()
    convert.to_gmsh(layout)
    node_count = len(gmsh.model.getPhysicalGroups(0))
    edge_count = len(gmsh.model.getPhysicalGroups(1))
    gmsh.finalize()

    assert node_count == layout.x.shape[0]
    assert edge_count == layout.cells.shape[0]


def test_to_pyvista(layout: Layout) -> None:
    plotter = convert.to_pyvista(layout)
    assert plotter is not None
