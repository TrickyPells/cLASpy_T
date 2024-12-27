# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cLASpy_T'
copyright = '2024, Xavier PELLERIN LE BAS'
author = 'Xavier PELLERIN LE BAS'
release = '0.3'
version = '0.3.3'

rst_epilog = """
.. |claspyt| replace:: **cLASpy_T**
.. |gui| replace:: :command:`cLASpy_GUI`
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The master toctree document.
master_doc = 'index'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = "default"
html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'
