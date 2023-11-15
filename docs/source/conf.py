# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# from importlib.metadata import version as get_version

root_dir= os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'harissa'
copyright = '2023, Ulysse Herbach'
author = 'Ulysse Herbach'
# release = get_version(project)
release = '3.0.7'
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_multiversion',
    'nbsphinx',
    # 'sphinx_gallery.gen_gallery',   
    'sphinx_gallery.load_style',
    'sphinx_copybutton',
    # 'myst-nb'
]

autosummary_generate = True
autosummary_ignore_module_all = False
# autosummary_imported_members = True

# sphinx_gallery_conf = {
#     'examples_dirs': '../../notebooks',   # path to your example scripts
#     'gallery_dirs': 'notebooks',  # path to where to save gallery generated output
# }

# source_suffix = {
#     '.rst': 'restructuredtext',
#     # '.txt': 'restructuredtext',
#     '.md': 'markdown',
# }

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_title = f'{project.capitalize()} documentation'
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "announcement": "<em>Under construction !</em>",
}
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/versioning.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        # 'sidebar/ethical-ads.html',
        'sidebar/scroll-end.html',
        # 'sidebar/variant-selector.html',
    ],
}


# -- Options for nbsphinx ---------------------------------------------------
nbsphinx_execute_arguments = [
  "--InlineBackend.figure_formats={'svg', 'pdf'}",
  "--InlineBackend.rc=figure.dpi=96"
]


#-- Options for Napoleon ----------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True


# -- Options for sphinx-copybutton ------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest
copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = '$ '

# -- Options for sphinx-multiversion
# https://holzhaus.github.io/sphinx-multiversion/master/index.html
smv_tag_whitelist = r'^(v|V)\d+\.\d+\.\d+$'
smv_branch_whitelist = None
smv_remote_whitelist = None
smv_released_pattern = r'^tags/.*$' 
