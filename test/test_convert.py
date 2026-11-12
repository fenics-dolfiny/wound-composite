from collections.abc import Callable

import gmsh
import pytest

from wound_composite import convert
from wound_composite.layout import Layout
from wound_composite.layout.templates import triad


@pytest.mark.parametrize("_layout", [triad])
def test_meshing(_layout: Callable[[], Layout]) -> None:
    gmsh.initialize()
    convert.to_gmsh(layout := _layout())
    node_count = len(gmsh.model.getPhysicalGroups(0))
    edge_count = len(gmsh.model.getPhysicalGroups(1))
    gmsh.finalize()

    assert node_count == layout.x.shape[0]
    assert edge_count == layout.cells.shape[0]
