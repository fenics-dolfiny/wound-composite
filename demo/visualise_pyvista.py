import os
from pathlib import Path

import pyvista as pv

import wound_composite.convert
from wound_composite.layout.templates import triad

pyvista_gui = True
show = (os.environ.get("CI") is None) and pyvista_gui

layout = triad()

# Create plotter (press 'i' to toggle indices, click to query data)
pv.OFF_SCREEN = pv.OFF_SCREEN or not show
plotter = wound_composite.convert.to_pyvista(layout)
if show:
    plotter.show()
else:
    plotter.screenshot(Path(__file__).parent / "triad.png")
