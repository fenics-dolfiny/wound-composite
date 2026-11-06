"""Sphinx configuration."""

project = "wound_composite"
author = "Paul T. Kühner, Michal Habera, Andreas Zilian"
copyright = f"2026, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

exclude_patterns = ["_build"]

# Fail (with -W) on any cross-reference that cannot be resolved.
nitpicky = True

myst_enable_extensions = ["dollarmath"]

autodoc_typehints = "description"
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "ufl": ("https://docs.fenicsproject.org/ufl/main", None),
}

html_theme = "furo"
html_title = project
