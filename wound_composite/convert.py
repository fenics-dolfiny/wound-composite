import gmsh
import numpy as np
import pyvista as pv
from numpy import typing as npt

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


def to_pyvista(
    layout: Layout, plotter: pv.Plotter = pv.Plotter(), string_width: float = 0.02
) -> pv.Plotter:
    labels = []
    radius_bushing = float(np.max(layout.offset_r))
    for vertex in range(layout.x.shape[0]):
        x = layout.x[vertex]
        direction = layout.orientation_array[vertex]

        offset_h = None
        for cell, (vertex_a, vertex_b) in enumerate(layout.cells):
            if vertex == vertex_a:
                offset_h = layout.offset_h[cell, 0]
            elif vertex == vertex_b:
                offset_h = layout.offset_h[cell, 1]
        assert offset_h is not None

        plotter.add_mesh(
            pv.Arrow(start=x, direction=direction, scale=radius_bushing), color="black"
        )
        plotter.add_mesh(
            pv.Tube(
                pointa=x - direction * offset_h,
                pointb=x + direction * offset_h,
                radius=radius_bushing,
            ),
            opacity=0.5,
        )
        labels.append(pv.Label(f"v{vertex}", x, name="label"))

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

        tube = pv.Tube(pointa=a, pointb=b, radius=string_width / 2, capping=True)
        plotter.add_mesh(tube, color="grey")
        labels.append(pv.Label(f"c{cell}", (a + b) / 2, name="label"))

    for label in labels:
        label.SetVisibility(False)
        plotter.add_actor(label, render=False)

    def toggle_indices() -> None:
        for label in labels:
            label.SetVisibility(not label.GetVisibility())
        plotter.render()

    plotter.add_key_event("i", toggle_indices)  # type: ignore

    return plotter
