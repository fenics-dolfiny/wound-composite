from textwrap import fill

import dolfiny.pyvista
import gmsh
import numpy as np
import pyvista as pv
from numpy import typing as npt
from vtkmodules.vtkRenderingCore import vtkCellPicker

from wound_composite.layout import Layout


def _addPoint(coords: npt.NDArray, search_radius: float = 1e-4, tag: int = -1) -> int:
    """Create GMSH point or return present tag.

    Based on implementation found at https://hpc.montefiore.ulg.ac.be/gmsh/gmsh/-/issues/1999.
    """
    pt = np.reshape(coords, 3)

    res = gmsh.model.occ.getEntitiesInBoundingBox(
        xmin=pt[0] - search_radius,
        ymin=pt[1] - search_radius,
        zmin=pt[2] - search_radius,
        xmax=pt[0] + search_radius,
        ymax=pt[1] + search_radius,
        zmax=pt[2] + search_radius,
        dim=0,
    )

    if res:
        assert len(res) == 1
        return res[-1][1]  # type: ignore

    return gmsh.model.occ.addPoint(*pt, tag=tag)  # type: ignore


def to_gmsh(layout: Layout, name: str = "resolved") -> gmsh.model:
    """Create gmsh geometry of GRAM layout.

    Arguments:
        layout: The layout to create geometry of
        name: Name by which the gmsh model can be identified

    Returns:
        GMSH model, new geometry is selected.

    """
    gmsh.model.add(name)

    bushing_count = layout.x.shape[0]  # TODO: make property
    # string_count = layout.cells.shape[1] # TODO: make property

    cylinders = []
    centers = []
    for i in range(bushing_count):
        x, y, z = layout.x[i]
        dx, dy, dz = layout.orientation_array[i]
        r = layout.offset_r.max()
        cyl = gmsh.model.occ.addCylinder(x - dx / 2, y - dy / 2, z - dz / 2, dx, dy, dz, r)
        cylinders.append(cyl)

        gmsh.model.occ.synchronize()

        centers.append(gmsh.model.occ.addPoint(x, y, z))

    strings = []
    line_points = []
    for cell, (vertex_a, vertex_b) in enumerate(layout.cells):
        orient_a = layout.orientation_array[vertex_a]
        orient_b = layout.orientation_array[vertex_b]

        a_to_b = layout.x[vertex_b] - layout.x[vertex_a]

        n_a_to_b_orient_a = np.cross(a_to_b, orient_a)
        n_a_to_b_orient_a /= np.linalg.norm(n_a_to_b_orient_a)

        a = layout.x[vertex_a].copy()
        a += layout.offset_h[cell, 0] * orient_a
        a += layout.offset_r[cell, 0] * n_a_to_b_orient_a

        n_a_to_b_orient_b = np.cross(a_to_b, orient_b)
        n_a_to_b_orient_b /= np.linalg.norm(n_a_to_b_orient_b)

        b = layout.x[vertex_b].copy()
        b += layout.offset_h[cell, 1] * orient_b
        b += layout.offset_r[cell, 1] * n_a_to_b_orient_b

        start = _addPoint(a)
        end = _addPoint(b)

        line_points += [start, end]

        line = gmsh.model.occ.addLine(start, end)

        strings.append(line)

    gmsh.model.occ.synchronize()

    frag_objs = [(3, c) for c in cylinders]
    frag_tools = [(0, c) for c in centers] + [(0, v) for v in np.unique(line_points)]
    frag_out, frag_map = gmsh.model.occ.fragment(frag_objs, frag_tools)
    gmsh.model.occ.synchronize()

    assert all(len(m) == 1 for m in frag_map)
    assert len(cylinders) == len([t for (d, t) in frag_out if d == 3])
    assert len(frag_map) == len(frag_objs) + len(frag_tools)

    # remap dim 0 entities
    for i in range(len(centers)):
        frag_idx = frag_tools.index((0, centers[i]))
        new_tag = frag_map[len(frag_objs) + frag_idx][0][1]
        centers[i] = new_tag

    # for i in range(len(line_points)):
    #     frag_idx = frag_tools.index((0, line_points[i]))
    #     new_tag = frag_map[len(frag_objs) + frag_idx][0][1]
    #     line_points[i] = new_tag

    for cell, line in enumerate(strings):
        pg_line = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, pg_line, f"string_{cell}")

    for i in range(bushing_count):
        pg_center = gmsh.model.addPhysicalGroup(0, [centers[i]])
        gmsh.model.setPhysicalName(0, pg_center, f"center_{i}")

        pg = gmsh.model.addPhysicalGroup(3, [cylinders[i]])
        gmsh.model.setPhysicalName(3, pg, f"bushing_{i}")

    # Force exactly 1 element for each string
    for tag in strings:
        gmsh.model.mesh.setTransfiniteCurve(tag, 2)

    return gmsh.model


def _string_ends(
    layout: Layout,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute both ends of every string, each of shape (-1, 3)."""
    a_to_b = layout.x[layout.cells[:, 1]] - layout.x[layout.cells[:, 0]]

    ends = []
    for end in range(2):
        orient = layout.orientation_array[layout.cells[:, end]]
        normal = np.cross(a_to_b, orient)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)

        point = layout.x[layout.cells[:, end]].copy()
        point += layout.offset_h[:, [end]] * orient
        point += layout.offset_r[:, [end]] * normal
        ends.append(point)

    return ends[0], ends[1]


def _bushing_envelopes(
    layout: Layout,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the envelope of the string ends attached to every bushing.

    Returns:
        Radius (largest radial offset) and axial span (smallest and largest
        axial offset) per bushing. A bushing without strings has radius 0
        and the empty span (inf, -inf).

    """
    bushing_count = layout.x.shape[0]
    radius = np.zeros(bushing_count)
    h_min = np.full(bushing_count, np.inf)
    h_max = np.full(bushing_count, -np.inf)

    np.maximum.at(radius, layout.cells, np.abs(layout.offset_r))
    np.minimum.at(h_min, layout.cells, layout.offset_h)
    np.maximum.at(h_max, layout.cells, layout.offset_h)

    return radius, h_min, h_max


def _segments(start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]) -> pv.PolyData:
    """Create one line per pair of start and end point."""
    count = start.shape[0]
    lines = np.column_stack([np.full(count, 2), np.arange(count), np.arange(count) + count])
    return pv.PolyData(np.vstack([start, end]), lines=lines.ravel())


def _tick_spacing(length: float) -> float:
    """Compute the tick spacing `vtkCubeAxesActor` uses for an axis length.

    Mirrors `vtkCubeAxesActor::AdjustTicksComputeRange`.
    """
    power = np.log10(length)
    if power != 0:
        power = np.copysign(abs(power) + 1e-9, power)
    if power < 0:
        power -= 1
    decade = 10.0 ** int(power)

    ticks = int(length / decade) + 1
    return float(decade / (5 if ticks <= 2 else 2 if ticks < 5 else 1))


def _tick_bounds(lower: float, upper: float) -> tuple[float, int]:
    """Extend an axis range downwards such that its ticks are round values.

    `vtkCubeAxesActor` places ticks at lower + i * spacing, which are
    multiples of the spacing only if the lower bound is. Extending the
    range may in turn change the spacing, hence the iteration.

    Returns:
        Extended lower bound and the number of ticks.

    """
    if not upper > lower:
        return lower, 0

    spacing = 0.0
    while (spacing_range := _tick_spacing(upper - lower)) != spacing:
        spacing = spacing_range
        lower = float(np.floor(lower / spacing + 1e-9) * spacing)

    return lower, int((upper - lower) / spacing + 1e-9) + 1


def to_pyvista(
    layout: Layout, plotter: pv.Plotter | None = None, string_width: float = 0.02
) -> pv.Plotter:
    """Create pyvista scene of GRAM layout.

    The scene is to scale and contains, per

    - string: a tube of diameter `string_width` between its two ends,
    - bushing: its centre, its axis (an arrow of fixed length pointing
      along the orientation) and the envelope of the string ends attached
      to it, i.e. the cylinder spanned by their largest radial and their
      smallest and largest axial offset. It is a disc if all strings end
      at the same axial offset, and omitted if no string ends off the axis.

    Indices of bushings (v) and strings (c) are labelled, press 'i' to
    toggle. Click on a string or bushing to display its data, click next
    to the layout to clear it. The bounding box carries the coordinate
    axes.

    Arguments:
        layout: The layout to visualise
        plotter: Plotter to add the scene to, it is used as is. If not
                 provided, one with the dolfiny theme is created.
        string_width: Diameter of the strings

    Returns:
        The plotter holding the scene.

    """
    themed = plotter is None
    if plotter is None:
        plotter = pv.Plotter(theme=dolfiny.pyvista.theme)
        plotter.enable_anti_aliasing("ssaa")
        plotter.enable_depth_peeling()  # type: ignore[call-arg]

    interactor = plotter.iren
    assert interactor is not None
    renderer = plotter.renderer
    font = plotter.theme.font
    line_width = max(1, plotter.window_size[1] // 1000)
    highlight = "orangered"

    x = layout.x
    orientation = layout.orientation_array
    bushing_count = x.shape[0]
    string_count = layout.cells.shape[0]

    # Strings
    a, b = _string_ends(layout)
    centre_lines = _segments(a, b)
    centre_lines.point_data["cell"] = np.tile(np.arange(string_count), 2)
    strings = centre_lines.tube(radius=string_width / 2, n_sides=32)
    strings.point_data["selected"] = np.zeros(strings.n_points)

    # Entities which can be selected, the selected one is highlighted. Its
    # points are flagged as cell data is not rendered by all VTK backends.
    selectable = {
        "cell": plotter.add_mesh(
            strings,
            scalars="selected",
            cmap=["grey", highlight],
            clim=[0, 1],
            show_scalar_bar=False,
            smooth_shading=True,
            name="strings",
        )
    }

    # Bushings
    radius, h_min, h_max = _bushing_envelopes(layout)
    envelopes = []
    for vertex in np.flatnonzero(radius > 0):
        centre = x[vertex] + (h_min[vertex] + h_max[vertex]) / 2 * orientation[vertex]
        if h_min[vertex] == h_max[vertex]:
            envelope = pv.Polygon(
                center=centre, radius=radius[vertex], normal=orientation[vertex], n_sides=128
            )
        else:
            envelope = pv.Cylinder(
                center=centre,
                direction=orientation[vertex],
                radius=radius[vertex],
                height=h_max[vertex] - h_min[vertex],
                resolution=128,
            )
        envelope.point_data["vertex"] = np.full(envelope.n_points, vertex)
        envelopes.append(envelope)

    if envelopes:
        bushings = pv.merge(envelopes, merge_points=False)
        bushings.point_data["selected"] = np.zeros(bushings.n_points)
        selectable["vertex"] = plotter.add_mesh(
            bushings,
            scalars="selected",
            cmap=[plotter.theme.color.hex_rgb, highlight],
            clim=[0, 1],
            show_scalar_bar=False,
            smooth_shading=True,
            opacity=0.5,
            name="bushings",
        )
        plotter.add_mesh(
            bushings.extract_feature_edges(), color="black", line_width=line_width, pickable=False
        )
        plotter.add_silhouette(bushings, color="black", line_width=line_width)

    # Axis of each bushing: starts at the centre, or the envelope if that
    # extends further, and ends in an arrow head beyond the envelope.
    arrow = float(radius.max()) if radius.max() > 0 else 5 * string_width
    axis_start = x + np.minimum(h_min, 0)[:, np.newaxis] * orientation
    axis_end = x + (np.maximum(h_max, 0) + arrow)[:, np.newaxis] * orientation
    heads = pv.merge(
        [
            pv.Cone(
                center=axis_end[vertex] - 0.2 * arrow * orientation[vertex],
                direction=orientation[vertex],
                height=0.4 * arrow,
                radius=0.12 * arrow,
                resolution=32,
            )
            for vertex in range(bushing_count)
        ]
    )
    plotter.add_mesh(
        _segments(axis_start, axis_end), color="black", line_width=line_width, pickable=False
    )
    plotter.add_mesh(heads, color="black", pickable=False)
    plotter.add_mesh(
        pv.PolyData(x),
        color="black",
        point_size=5 * line_width,
        render_points_as_spheres=True,
        pickable=False,
    )

    # Index labels
    labels = [
        plotter.add_point_labels(
            x,
            [f"v{vertex}" for vertex in range(bushing_count)],
            always_visible=True,
            show_points=False,
            shape_color="white",
            margin=0,
            name="vertex",  # actor is named vertex-labels
        ),
        plotter.add_point_labels(
            (a + b) / 2,
            [f"c{cell}" for cell in range(string_count)],
            bold=False,
            always_visible=True,
            show_points=False,
            shape_color="white",
            margin=0,
            name="cell",  # actor is named cell-labels
        ),
    ]

    def toggle_indices() -> None:
        for actor in labels:
            actor.SetVisibility(not actor.GetVisibility())
        plotter.render()

    plotter.add_key_event("i", toggle_indices)  # type: ignore

    # Coordinate axes
    bounds = np.asarray(renderer.bounds).reshape(3, 2)
    ticks = [_tick_bounds(lower, upper) for lower, upper in bounds]
    bounds[:, 0] = [lower for lower, _ in ticks]
    cube_axes = plotter.show_bounds(  # type: ignore[call-arg]
        bounds=bounds.ravel(),
        n_xlabels=ticks[0][1],
        n_ylabels=ticks[1][1],
        n_zlabels=ticks[2][1],
        xtitle="x",
        ytitle="y",
        ztitle="z",
        fmt="%.12g",  # exact for round ticks, unlike the format of the theme
        grid="back",
        location="outer",
        ticks="outside",
    )
    # Text of the axes is not set in the font of the theme, match its size.
    cube_axes.SetScreenSize(font.size / 5)
    cube_axes.title_offset = (1.5 * font.size, 1.5 * font.size)
    # The lowest tick of the z-axis is the first tick of the x- or y-axis
    # it meets, where the labels of both would overlap.
    cube_axes.SetAxisLabels(2, pv.convert_string_array(np.array(["", *cube_axes.z_labels[1:]])))

    # Query by click
    length = np.linalg.norm(b - a, axis=1)
    tolerance = 1e-12 * float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))

    def number(value: float) -> str:
        return f"{0.0 if abs(value) < tolerance else value:.6g}"

    def vector(values: npt.NDArray[np.float64]) -> str:
        return f"({', '.join(number(value) for value in values)})"

    def describe(kind: str, index: int) -> str:
        if kind == "cell":
            vertex_a, vertex_b = layout.cells[index]
            offset_h, offset_r = layout.offset_h[index], layout.offset_r[index]
            return "\n".join(
                [
                    f"string c{index}: v{vertex_a} -> v{vertex_b}",
                    f"start     {vector(a[index])}",
                    f"          offset_h {number(offset_h[0])}, offset_r {number(offset_r[0])}",
                    f"end       {vector(b[index])}",
                    f"          offset_h {number(offset_h[1])}, offset_r {number(offset_r[1])}",
                    f"length    {number(length[index])}",
                ]
            )

        cells = np.flatnonzero(np.any(layout.cells == index, axis=1))
        attached = fill(" ".join(f"c{cell}" for cell in cells), 50, subsequent_indent=" " * 13)
        return "\n".join(
            [
                f"bushing v{index}",
                f"centre       {vector(x[index])}",
                f"orientation  {vector(orientation[index])}",
                f"radius       {number(radius[index])}",
                f"axial span   {number(h_min[index])} .. {number(h_max[index])}",
                f"strings      {attached}",
            ]
        )

    info = plotter.add_text(
        "",
        position=(0.02, 0.98),
        viewport=True,
        font_size=font.size // 2,
        font=font.family,  # type: ignore[arg-type]
        name="query",
    )
    info.prop.justification_vertical = "top"
    info.prop.SetLineSpacing(1.3)

    picker = vtkCellPicker()
    pressed = [0, 0]

    def press(*_: object) -> None:
        pressed[:] = interactor.get_event_position()

    def release(*_: object) -> None:
        position = interactor.get_event_position()
        if max(abs(position[0] - pressed[0]), abs(position[1] - pressed[1])) > 2:
            return  # dragged, not clicked

        picker.Pick(*position, 0, renderer)
        text = ""
        for kind, actor in selectable.items():
            data = actor.mapper.dataset
            selected = data.point_data["selected"]
            selected[:] = 0
            if picker.GetActor() is actor:
                index = int(data.point_data[kind][picker.GetPointId()])
                selected[data.point_data[kind] == index] = 1
                text = describe(kind, index)
            data.Modified()

        info.input = text
        plotter.render()

    interactor.add_observer("LeftButtonPressEvent", press)
    interactor.add_observer("LeftButtonReleaseEvent", release)

    if themed:
        plotter.view_isometric()  # type: ignore[call-arg]
        plotter.camera.zoom(0.85)  # room for the text of the axes

    return plotter
