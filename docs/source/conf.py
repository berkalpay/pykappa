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

html_theme_options = {
    "github_user": "berkalpay",
    "github_repo": "pykappa",
    "github_button": False,
    "github_banner": True,
}
html_static_path = ["_static"]
html_title = "PyKappa"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_css_files = ["custom.css"]
