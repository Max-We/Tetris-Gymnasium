# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Tetris Gymnasium"
copyright = "2024, Maxmilian Weichart"
author = "Maxmilian Weichart"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_title = "Tetris Gymnasium"
html_baseurl = "https://max-we.github.io/Tetris-Gymnasium"
html_copy_source = False
# html_favicon = "_static/img/favicon.png"
html_theme_options = {
    # "light_logo": "img/gymnasium_black.svg",
    # "dark_logo": "img/gymnasium_white.svg",
    "description": "A high performance, high customization Tetris environments for Gymnasium.",
    #     "image": "_static/logo.png",
    "versioning": True,
    "source_repository": "https://github.com/Max-We/Tetris-Gymnasium",
    "source_branch": "main",
    "source_directory": "docs/",
    "announcement": "Tetris Gymnasium is under early development!",
}
