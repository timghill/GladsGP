# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GladsGP'
copyright = '2024, Tim Hill'
author = 'Tim Hill'
release = '2024.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'nbsphinx',
    'nbsphinx_link',
]

napoleon_use_admonition_for_examples = True

templates_path = ['_templates']
exclude_patterns = []


source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/timghill/GladsGP",
    "use_repository_button": True,
}

# html_title = "My site title"

# This code adds a banner to notify the viewer of pages that
# were automatically generated using nbsphinx and points the
# viewer to the source *.ipynb on GitHub
nbsphinx_prolog = (
r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'] %}
{% else %}
{% set docpath = "source/" + env.doc2path(env.docname, base=None) %}
{% endif %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
        This page was generated from `{{ docpath }}`__.

    __ https://github.com/timghill/GladsGP/tree/main/docs/""" + r"{{ docpath }}"
)
