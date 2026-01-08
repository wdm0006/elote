# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath("."))))


# -- Project information -----------------------------------------------------

project = "elote"
copyright = "2020, Will McGinnis"
author = "Will McGinnis"

# The full version, including alpha/beta/rc tags
try:
    # Try to get version from importlib.metadata (Python 3.8+)
    from importlib.metadata import version as get_version

    release = get_version("elote")
except ImportError:
    # Fallback for older Python versions
    try:
        import pkg_resources

        release = pkg_resources.get_distribution("elote").version
    except Exception:  # Replace bare except with specific exception type
        # Hardcoded fallback
        release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension modules here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinxcontrib.googleanalytics",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "wabi_sphinx_theme"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "wabi_sphinx_theme.pygments_style.WabiStyle"

# Google Analytics configuration
googleanalytics_id = "G-Z43R9PWW0B"
googleanalytics_enabled = True

# Theme options
html_theme_options = {
    "site_title": "McGinnis, Will",
    "site_url": "https://mcginniscommawill.com",
    "docs_base_url": "https://elote.mcginniscommawill.com",
    "nav_links": [
        {"label": "Guides", "url": "https://mcginniscommawill.com/guides/"},
        {"label": "Topics", "url": "https://mcginniscommawill.com/topics/"},
        {"label": "Blog", "url": "https://mcginniscommawill.com/posts/"},
        {"label": "About", "url": "https://mcginniscommawill.com/about/"},
        {"label": "Free Coffee", "url": "https://mcginniscommawill.com/coffee/"},
        {"label": "OSS", "url": "https://mcginniscommawill.com/oss/"},
    ],
    "nav_show_docs_link": True,
    "nav_docs_label": "Docs",
    "show_breadcrumbs": True,
    "show_home_breadcrumb": True,
    "twitter_site": "@willmcginniser",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "custom.css",
]

html_extra_path = ["CNAME"]

autoclass_content = "both"
