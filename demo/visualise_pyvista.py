import os
from pathlib import Path

import wound_composite.convert
from wound_composite.layout.templates import triad

pyvista_gui = True
show = (os.environ.get("CI") is None) and pyvista_gui

layout = triad()

# Create plotter (press 'i' for index information)
plotter = wound_composite.convert.to_pyvista(layout)
if show:
    plotter.show()
else:
    plotter.save_graphic(Path(__file__).parent / "triad.pdf")
