# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re 
import sys
from importlib.metadata import version as get_version

cmd_name = os.path.basename(os.environ['_'])
is_multi_version_sub_process = (
    cmd_name == 'sphinx-multiversion' 
    and cmd_name != os.path.basename(sys.argv[0])
)

if is_multi_version_sub_process:
    sys.path[0] =  os.path.join(sys.path[0], 'src')

    def root_index_content(redirect_version):
        v_index = f'{redirect_version}/index.html' 
        relative_url = f'./{v_index}'
        absolute_url = f'harissa-framework.github.io/harissa/{v_index}'

        return f'''<!DOCTYPE html>
<html>
    <head>
        <title>Redirecting to version {redirect_version}</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url={relative_url}">
        <link rel="canonical" href="{absolute_url}">
    </head>
</html>
'''

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
# autosummary_filename_map = {
#     'harissa.utils.plot_network': 'harissa.utils.plot_network_m'
# }
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

if is_multi_version_sub_process:
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

tag_whitelist = r'^v'
if active_versions:
    tag_whitelist += r'('
    for i, active_version in enumerate(active_versions):
        active_version_escaped = active_version.replace(r'.', r'\.')
        if i > 0:
            tag_whitelist += rf'|{active_version_escaped}'
        else:
            tag_whitelist += rf'{active_version_escaped}'
    tag_whitelist += r')'

    # Choose here the version latest version
    smv_latest_version = active_versions[-1]

    if is_multi_version_sub_process:
        # Choose here the version to redirect when landing on the root index
        redirect_version = smv_latest_version
        def generate_root_index(app, exception):
            if exception is not None:
                return
            
            root_index = os.path.join(app.outdir, '..' , 'index.html')
                
            print('\033[1mBuilding root index.html\033[0m')

            with open(root_index, 'w') as file:
                file.write(root_index_content(redirect_version))   

            print(f'{root_index} generated.\n' 
                f'It redirects to version {redirect_version}')
        
        def setup(app):
            current_version = os.path.basename(app.outdir)
            
            # fallback redirection
            if current_version == active_versions[0]:
                root_index = os.path.join(app.outdir, '..' , 'index.html')
                with open(root_index, 'w') as file:
                    file.write(root_index_content(active_versions[0]))

            for active_version in active_versions:
                if (redirect_version == active_version 
                    and active_version == current_version):
                    app.connect('build-finished', generate_root_index)
                    break

tag_whitelist += r'$'

smv_tag_whitelist = tag_whitelist
smv_branch_whitelist = r'^main$'
smv_remote_whitelist = r'^$'
smv_released_pattern = r'^tags/v\d+(\.\d+)*$'