# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as _version, PackageNotFoundError

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -------------------------------------------------------

project = 'fSTG Toolkit'
copyright = '2025, ICube (University of Strasbourg – CNRS)'
author = 'Julien PONTABRY'

try:
    release = _version('fSTG-Toolkit')
except PackageNotFoundError:
    release = 'unknown'

version = '.'.join(release.split('.')[:2])

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx_click',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- MyST configuration --------------------------------------------------------

myst_enable_extensions = ['colon_fence']

# -- autodoc configuration -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

autosummary_generate = True

# -- Napoleon configuration (NumPy docstrings) ---------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- Intersphinx configuration -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'networkx': ('https://networkx.org/documentation/stable', None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '_static/images/fSTG_logo.svg'

html_theme_options = {
    'source_repository': 'https://github.com/julienpontabry/fstg_toolkit',
    'source_branch': 'main',
    'source_directory': 'docs/',
}
