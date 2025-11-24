# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyKappa"
copyright = "2025, PyKappa Developers"
author = "PyKappa Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": True}
autoclass_content = "class"
napoleon_google_docstring = True
nbsphinx_execute = "always"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "secondary_sidebar_items": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/berkalpay/pykappa",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}
html_static_path = ["_static"]
html_title = "PyKappa"
