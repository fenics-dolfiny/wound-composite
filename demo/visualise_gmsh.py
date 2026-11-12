import os

import gmsh

import wound_composite.convert
from wound_composite.layout.templates import triad

gmsh_gui = True
show = (os.environ.get("CI") is None) and gmsh_gui

layout = triad()

gmsh.initialize()

# Geometry (non-meshed)
wound_composite.convert.to_gmsh(layout)
if show:
    gmsh.fltk.run()

# Meshing
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
gmsh.model.mesh.generate(3)
gmsh.model.geo.synchronize()
if show:
    gmsh.fltk.run()
