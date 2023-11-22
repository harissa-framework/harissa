# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re 
import sys
from importlib.metadata import version as get_version

is_multi_version = os.path.basename(os.environ['_']) == 'sphinx-multiversion'
is_sub_process = is_multi_version and os.getppid() + 1 != os.getpid()

if is_sub_process:
    sys.path[0] =  os.path.join(sys.path[0], 'src')
    output_dir = os.path.dirname(sys.argv[-1])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Harissa'
copyright = '2023, Ulysse Herbach'
author = 'Ulysse Herbach'
# harissa needs to be installed (at least in editable mode)
version = get_version(project.lower())
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.coverage',
    'sphinx_multiversion',
    'nbsphinx',
    # 'sphinx_gallery.gen_gallery',   
    'sphinx_gallery.load_style',
    'sphinx_copybutton',
    # 'myst-nb'
]

# coverage_show_missing_items = True
# coverage_statistics_to_stdout = True

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

if is_multi_version:
    html_title = f'{project} documentation'
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
pattern = re.compile(
    r'^(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$'
)
active_versions = [
    '3.0.7',
    # '3.0.8',
    # '4.0.0',
]
active_versions = tuple(
    filter(lambda v: re.match(pattern, v), active_versions)
)

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

    # Choose here the version latest version
    smv_latest_version = active_versions[-1]
    # Choose here the version to redirect when landing on the root index
    redirect_version = smv_latest_version
    if is_sub_process:
        os.makedirs(output_dir, exist_ok=True)
        root_index = os.path.join(output_dir, 'index.html')
        if not os.path.exists(root_index):
            print('\033[1mBuilding root index.html\033[0m')

            version_index=f'{redirect_version}/index.html' 
            relative_url=f'./{version_index}'
            absolute_url=f'harissa-framework.github.io/harissa/{version_index}'

            with open(root_index, 'w') as file:
                file.write(f'''<!DOCTYPE html>
<html>
    <head>
        <title>Redirecting to version {smv_latest_version}</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url={relative_url}">
        <link rel="canonical" href="{absolute_url}">
    </head>
</html>
''')   

            print(f'{root_index} generated.\n' 
                f'It redirects to version {smv_latest_version}')
tag_whitelist += r'$'

smv_tag_whitelist = tag_whitelist
smv_branch_whitelist = r'^$'
smv_remote_whitelist = r'^$'
smv_released_pattern = r'^tags/\d+(\.\d+)*$'