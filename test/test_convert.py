from collections.abc import Callable

import gmsh
import numpy as np
import pytest
import pyvista as pv

from wound_composite import convert
from wound_composite.layout import Layout
from wound_composite.layout.templates import triad


def _event(plotter: pv.Plotter, event: str, x: int = 0, y: int = 0, key: str = "") -> None:
    """Invoke an event of the interactor."""
    assert plotter.iren is not None
    plotter.iren.interactor.SetEventInformation(x, y)
    plotter.iren.interactor.SetKeySym(key)
    plotter.iren.interactor.InvokeEvent(event)


def _click(plotter: pv.Plotter, point: np.ndarray, drag: int = 0) -> str:
    """Click on the projection of a point and return the displayed data."""
    plotter.renderer.SetWorldPoint(*point, 1.0)
    plotter.renderer.WorldToDisplay()
    x, y, _ = np.rint(plotter.renderer.GetDisplayPoint()).astype(int)

    _event(plotter, "LeftButtonPressEvent", x, y)
    _event(plotter, "LeftButtonReleaseEvent", x + drag, y)
    return str(plotter.actors["query"].input)


def _selected(plotter: pv.Plotter, name: str, kind: str) -> list[int]:
    """Return indices of the highlighted entities of an actor."""
    data = plotter.actors[name].mapper.dataset
    selected = data.point_data[kind][data.point_data["selected"] == 1]
    return [int(index) for index in np.unique(selected)]


@pytest.mark.parametrize("_layout", [triad])
def test_to_gmsh(_layout: Callable[[], Layout]) -> None:
    gmsh.initialize()
    convert.to_gmsh(layout := _layout())
    node_count = len(gmsh.model.getPhysicalGroups(0))
    edge_count = len(gmsh.model.getPhysicalGroups(1))
    gmsh.finalize()

    assert node_count == layout.x.shape[0]
    assert edge_count == layout.cells.shape[0]


@pytest.mark.parametrize("_layout", [triad])
def test_string_ends(_layout: Callable[[], Layout]) -> None:
    layout = _layout()
    a, b = convert._string_ends(layout)

    # Reference configuration of the mesh is x + offset
    offset = layout.offset.x.array.reshape(-1, 3)
    dofs = layout.offset.function_space.dofmap.list
    assert np.allclose(a, layout.x[layout.cells[:, 0]] + offset[dofs[:, 0]])
    assert np.allclose(b, layout.x[layout.cells[:, 1]] + offset[dofs[:, 1]])


def test_bushing_envelopes() -> None:
    radius, h_min, h_max = convert._bushing_envelopes(triad())

    assert np.allclose(radius, 0.2)
    assert np.allclose(h_min, -0.5)
    assert np.allclose(h_max, [0.5, -0.5, 0.5])  # all strings end on one side of v1


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((-0.2, 3.2), (-0.5, 8)),
        ((-1.45, 0.7), (-1.5, 5)),
        ((0.0, 1.0), (0.0, 6)),
        ((0.03, 0.1), (0.03, 8)),
        ((-130.0, 870.0), (-200.0, 6)),
        ((1.0, 1.0), (1.0, 0)),
    ],
)
def test_tick_bounds(bounds: tuple[float, float], expected: tuple[float, int]) -> None:
    lower, count = convert._tick_bounds(*bounds)

    assert np.isclose(lower, expected[0])
    assert count == expected[1]


@pytest.mark.parametrize("_layout", [triad])
@pytest.mark.parametrize("scale", [0.03, 1.0, 700.0])
def test_to_pyvista(
    _layout: Callable[[], Layout], scale: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pv, "OFF_SCREEN", True)

    layout = _layout()
    layout = Layout(
        scale * layout.x,
        layout.cells,
        layout.orientation_array,
        scale * layout.offset_h,
        scale * layout.offset_r,
    )
    plotter = convert.to_pyvista(layout, string_width=scale * 0.02)

    # One tube per string, one envelope per bushing
    strings = plotter.actors["strings"].mapper.dataset
    bushings = plotter.actors["bushings"].mapper.dataset
    assert np.unique(strings.point_data["cell"]).size == layout.cells.shape[0]
    assert np.unique(bushings.point_data["vertex"]).size == layout.x.shape[0]

    # Ticks of the coordinate axes are multiples of their spacing
    axes = plotter.renderer.cube_axes_actor
    assert axes is not None
    for labels in (axes.x_labels, axes.y_labels, axes.z_labels[1:]):
        values = np.array([float(label) for label in labels])
        spacing = np.diff(values)
        assert np.allclose(spacing, spacing[0])
        assert np.allclose(values / spacing[0], np.rint(values / spacing[0]))
        assert labels == [f"{value:g}" for value in values]

    a, b = convert._string_ends(layout)
    initial = np.array(plotter.screenshot())

    # Click on string
    text = _click(plotter, (a[5] + b[5]) / 2)
    assert text.startswith("string c5: v0 -> v2")
    assert _selected(plotter, "strings", "cell") == [5]
    assert _selected(plotter, "bushings", "vertex") == []
    assert not np.array_equal(np.array(plotter.screenshot()), initial)

    # Dragging does not change the selection
    assert _click(plotter, layout.x[1], drag=10) == text

    # Click on bushing, v1 is a disc below its centre
    text = _click(plotter, layout.x[1] - scale * 0.5 * layout.orientation_array[1])
    assert text.startswith("bushing v1")
    assert "strings      c0 c1 c2 c3 c10 c11 c12 c13" in text
    assert _selected(plotter, "strings", "cell") == []
    assert _selected(plotter, "bushings", "vertex") == [1]

    # Click next to the layout, on a corner of the bounding box
    assert _click(plotter, np.asarray(axes.bounds)[[1, 3, 4]]) == ""
    assert _selected(plotter, "bushings", "vertex") == []

    # Toggle index labels
    for visible in (False, True):
        _event(plotter, "KeyPressEvent", key="i")
        assert plotter.actors["vertex-labels"].GetVisibility() == visible
        assert plotter.actors["cell-labels"].GetVisibility() == visible

    plotter.close()


def test_to_pyvista_no_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pv, "OFF_SCREEN", True)

    # Single string ending in the centres, v2 holds no string
    layout = Layout(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
    )
    plotter = convert.to_pyvista(layout, plotter=pv.Plotter())

    assert "strings" in plotter.actors
    assert "bushings" not in plotter.actors
    plotter.screenshot()
    assert _click(plotter, np.array([0.5, 0, 0])).startswith("string c0: v0 -> v1")

    plotter.close()
