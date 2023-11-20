# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re 
from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Harissa'
copyright = '2023, Ulysse Herbach'
author = 'Ulysse Herbach'
release = get_version(project.lower())
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

if 'HTML_TITLE' in os.environ:
    html_title = os.environ['HTML_TITLE']
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
# html_theme_options = {
#     "announcement": "<em>Under Construction !</em>",
# }
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
html_copy_source = False


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

# Convention for version number https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
# [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
pattern=re.compile(r'^(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$')
active_versions = [
    '3.0.7',
    '4.0.0'
]
active_versions=tuple(filter(lambda v: re.match(pattern, v), active_versions))

# -- Options for sphinx-multiversion
# https://holzhaus.github.io/sphinx-multiversion/master/index.html

tag_whitelist = r'^'
if active_versions:
    tag_whitelist += r'('
    for i, active_version in enumerate(active_versions):
        active_version = active_version.replace(r'.', r'\.')
        if i > 0:
            tag_whitelist += rf'|{active_version}'
        else:
            tag_whitelist += rf'{active_version}'
    tag_whitelist += r')'
tag_whitelist += r'$'

smv_tag_whitelist = tag_whitelist
smv_branch_whitelist = r'^$'
smv_remote_whitelist = r'^$'
smv_released_pattern = r'^tags/\d+(\.\d+)*$'
smv_latest_version = active_versions[0]