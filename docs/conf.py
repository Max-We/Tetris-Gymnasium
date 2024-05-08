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
    # "announcement": "Tetris Gymnasium is under early development!",
}

# Autodoc
autoclass_content = "both"
autodoc_preserve_defaults = True


# This function removes the content before the parameters in the __init__ function.
# This content is often not useful for the website documentation as it replicates
# the class docstring.
def remove_lines_before_parameters(app, what, name, obj, options, lines):
    if what == "class":
        # ":param" represents args values
        first_idx_to_keep = next(
            (i for i, line in enumerate(lines) if line.startswith(":param")), 0
        )
        lines[:] = lines[first_idx_to_keep:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_lines_before_parameters)
