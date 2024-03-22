# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re 
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
from pathlib import Path
from shutil import copytree, rmtree, ignore_patterns

from sphinxcontrib.collections.drivers import Driver
from sphinxcontrib.collections.api import register_driver

project_dir = Path(__file__).parent.parent.parent
tmp_dir = project_dir

cmd_name = Path(os.environ['_']).name
is_multi_version_sub_process = (
    cmd_name == 'sphinx-multiversion' 
    and cmd_name != Path(sys.argv[0]).name
)


if is_multi_version_sub_process:
    tmp_dir = Path(sys.argv[-2]).parent.parent
    sys.path[0] =  str(tmp_dir / 'src')
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
    
use_nbsphinx_link = (tmp_dir / 'docs' / 'source' / 'notebooks').is_dir()
print(f'{use_nbsphinx_link=}')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Harissa'
copyright = '2023, Ulysse Herbach'
author = 'Ulysse Herbach'
# harissa needs to be installed (at least in editable mode)
import harissa

if hasattr(harissa, '__version__'):
    version = harissa.__version__
else:
    try:
        version = get_version(project.lower())
    except PackageNotFoundError:
        version = 'unknown version'

release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.coverage',
    'sphinx_multiversion',
    'myst_parser',
    'nbsphinx',
    # 'sphinx_gallery.gen_gallery',   
    'sphinx_gallery.load_style',
    'sphinx_copybutton',
    # 'myst-nb'
]

myst_enable_extensions = [
    # 'amsmath',
    'dollarmath',
    'colon_fence',
    'strikethrough',
    'tasklist',
]

if use_nbsphinx_link:
    extensions.append('nbsphinx_link')
else:
    extensions.append('sphinxcontrib.collections')

    # -- Options for sphinx collections output -------------------------------
    # https://sphinx-collections.readthedocs.io/en/latest/

    class CopyFolderOnly(Driver):
        def copy_only(self, path, names):
            patterns = self.config.get('only', [])
            return set(names) - ignore_patterns(*patterns)(path, names)
        
        def run(self):
            # override it to be able to copy in the tmp_dir (multiversion)
            # rel_dir = Path(self.config['target']).relative_to(project_dir)
            # self.config['target'] = str(tmp_dir / rel_dir)

            self.info(f'is sub process: {is_multi_version_sub_process}')

            self.info(
                (f'Copy folder {self.config["source"]} '
                f'into {self.config["target"]} ...')
            )

            if not Path(self.config['source']).exists():
                self.error(f'Source {self.config["source"]} does not exist')
                return

            try:
                copytree(
                    self.config['source'], 
                    self.config['target'], 
                    ignore=self.copy_only,
                    dirs_exist_ok=True # for sphinx-autobuild
                )
            except IOError as e:
                self.error("Problems during copying folder.", e)

        def clean(self):
            try:
                rmtree(self.config['target'])
                self.info(f'Folder deleted: {self.config["target"]}')
            except FileNotFoundError:
                pass  # Already cleaned? I'm okay with it.
            except IOError as e:
                self.error(f'Problems during cleaning for collection {self.config["name"]}', e)

    register_driver('copy_folder_only', CopyFolderOnly)

    # coverage_show_missing_items = True
    # coverage_statistics_to_stdout = True

    collections = {
        'notebooks' : {
            'driver': 'copy_folder_only',
            'source': str(tmp_dir / 'notebooks'),
            # 'target': str(tmp_dir/'docs'/'source'/'_collections'/'notebooks'),
            'only': ['*.ipynb'],
        }
    }
    collections_target = str(tmp_dir/'docs'/'source'/'_collections')

# -- Options for autodoc output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_typehints = 'description'
# autoclass_content = 'both'
# autodoc_class_signature = 'separated'

autosummary_generate = True
autosummary_ignore_module_all = False
# autosummary_imported_members = True

# sphinx_gallery_conf = {
#     'examples_dirs': '../../notebooks',   # path to your example scripts
#     'gallery_dirs': 'notebooks',  # path to where to save gallery generated output
# }

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

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
copybutton_exclude = '.linenos, .gp, .go, .o'
# copybutton_prompt_text = r"[$!]\s*"
# copybutton_prompt_is_regexp = True

# Convention for version number https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
# [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
pattern = re.compile(
    r'^v(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$'
)
active_versions = [
    # 'v3.0.7',
    'v3.0.8',
    # 'v4.0.0',
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
            current_version = Path(app.outdir).name
            # hack for sphinx collections and multiversion
            if not use_nbsphinx_link:
                app.confdir = tmp_dir /'docs'/'source'
            
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