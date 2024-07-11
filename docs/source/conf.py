# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re 
import sys
import sysconfig
from importlib.metadata import version as get_version, PackageNotFoundError
from pathlib import Path
from shutil import copytree, rmtree, ignore_patterns
import json
import venv
import subprocess
import shlex

from sphinxcontrib.collections.drivers import Driver
from sphinxcontrib.collections.api import register_driver

import sphinx

# -------------- Utility functions and class -----------------------

logger = sphinx.util.logging.getLogger(__name__)

# Convention for version number https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
# [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
PYPI_VERSION_PATTERN=r'(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?'
pattern = re.compile(f'^v{PYPI_VERSION_PATTERN}$')
switcher_filename = 'switcher.json' 
conf_dir = Path(__file__).parent 
with open(conf_dir / switcher_filename) as fp:
    switcher_data = list(filter(
        lambda data: re.match(pattern, data['version']), 
        json.load(fp)
    ))

def tag_whitelist():
    whitelist = r'^'
    if switcher_data:
        whitelist += r'('
        for i, data in enumerate(switcher_data):
            version = data['version'].replace(r'.', r'\.')
            if i > 0:
                whitelist += f'|{version}'
            else:
                whitelist += version
        whitelist += r')'

    whitelist += r'$'

    return whitelist

def to_sem_ver(version):
    semver = re.sub(r'(\d+)(a|b|rc)(\d+)', r'\g<1>-pre.\g<3>', version)
    semver = re.sub(r'\.post(\d+)', r'+\g<1>', semver)
    return re.sub(r'\.dev(\d+)', r'+build.\g<1>', semver)

def root_index_content(baseurl):
    index = 'stable/index.html'

    return f'''<!DOCTYPE html>
<html>
    <head>
        <title>Redirecting to stable version</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url=./{index}">
        <link rel="canonical" href="{baseurl}{index}">
    </head>
</html>
'''

def root_sitemap_index_content(baseurl, versions):
    sitemaps = map(
        lambda version: f'{baseurl}{version}/sitemap.xml',
        versions
    )
    sitemaps_str = "".join(map(
        lambda sitemap: f'<sitemap><loc>{sitemap}</loc></sitemap>',
        sitemaps
    ))

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{sitemaps_str}
</sitemapindex>
'''

class CopyFolderOnly(Driver):
    def copy_only(self, path, names):
        patterns = self.config.get('only', [])
        return set(names) - ignore_patterns(*patterns)(path, names)
    
    def run(self):
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
                dirs_exist_ok=True
            )
        except IOError as e:
            self.error("Problems during copying folder.", e)

    def clean(self):
        try:
            # rmtree(self.config['target'])
            target = Path(self.config['target'])
            patterns = self.config.get('only', [])
            names = list(map(lambda p: p.name, target.iterdir()))
            delete_only = ignore_patterns(*patterns)(target, names)
            for name in delete_only:
                path = target / name
                path.unlink()
                self.info(f'File deleted: {path}')
        except FileNotFoundError:
            pass  # Already cleaned? I'm okay with it.
        except IOError as e:
            error_msg = 'Problems during cleaning for collection '
            error_msg = error_msg + self.config['name']
            self.error(error_msg, e)

def reset_conf_dir(conf_dir):
    def wrapper(app, config):
        app.confdir = conf_dir

    return wrapper

if (conf_dir.parent.parent / '.git').is_dir():
    current_branch = subprocess.run(
        shlex.split('git rev-parse --abbrev-ref HEAD'),
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()
else:
    current_branch = 'main'

module_name = 'harissa'
try:
    current_branch_version = f'v{get_version(module_name)}'
except PackageNotFoundError:
    raise RuntimeError('harissa must be installed for the autodoc to work.')

if current_branch_version in map(lambda data: data['version'], switcher_data):
    current_branch = ''

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = module_name.capitalize()
copyright = '2023, Ulysse Herbach'
author = 'Ulysse Herbach'

version = current_branch_version[1:]
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.coverage',
    'sphinx_multiversion',
    'sphinx_copybutton',
    'sphinx_sitemap'
]

# -- Options for autodoc output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_typehints = 'description'
# autoclass_content = 'both'
# autodoc_class_signature = 'separated'

autosummary_generate = True
autosummary_ignore_module_all = False
# autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = f'https://harissa-framework.github.io/{module_name}/'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': f'https://github.com/harissa-framework/{module_name}',
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        }
   ],
   'pygments_light_style': 'default',
   'pygments_dark_style': 'material',
}
html_copy_source = False


# -- Options for nbsphinx ---------------------------------------------------
nbsphinx_execute_arguments = [
  "--InlineBackend.figure_formats={'svg', 'pdf'}",
  "--InlineBackend.rc=figure.dpi=96"
]
nbsphinx_execute = 'never'

nb_execution_mode = 'off'

# -- Options for myst-nb ----------------------------------------------------

nb_number_source_lines = True

# myst_heading_anchors = 3

myst_enable_extensions = [
    # 'amsmath',
    'dollarmath',
    'colon_fence',
    'strikethrough',
    'tasklist',
]

myst_fence_as_directive = {'mermaid'}
mermaid_version ="10.9.1"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme: 'neutral', legacyMathML: true});"
# mermaid_d3_zoom = True

#-- Options for Napoleon ----------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for sphinx-copybutton ------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest
copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = r'[$!]\s*'
copybutton_prompt_is_regexp = True

# -- Options for sphinx-multiversion
# https://holzhaus.github.io/sphinx-multiversion/master/index.html

smv_tag_whitelist = tag_whitelist()
smv_branch_whitelist = rf'^{current_branch}$'
smv_remote_whitelist = r'^$'

sitemap_url_scheme = "{link}"

def setup_collections(app, config):
    doc_src_dir = Path(app.srcdir)
    # -- Options for sphinx collections output -------------------------------
    # https://sphinx-collections.readthedocs.io/en/latest/
    config.collections = {
        'notebooks' : {
            'driver': 'copy_folder_only',
            'source': str(doc_src_dir.parent.parent / 'notebooks'),
            'only': ['*.ipynb'],
        },
        'benchmark' : {
            'driver': 'copy_folder_only',
            'source': str(doc_src_dir.parent.parent / 'benchmark' / 'notebooks'),
            'only': ['*.ipynb'],
        }
    }
    config.collections_target = str(doc_src_dir)
    # hack for sphinx collections and multiversion
    app.confdir = app.srcdir

def setup_multi_version(app, config):
    current_version = config.smv_current_version
    if current_version:
        tmp_project = Path(app.srcdir).parent.parent
        # Create a fresh git repo for setuptool-scm to retrieve dummy version
        git_cmds = [
            'git init', 
            'git config --local user.name sphinx-mv',
            'git config --local user.email sphinx-mv@example.com',
            'git add .',
            'git commit -m v0.0.1',
            'git tag v0.0.1'
        ]
        for cmd in git_cmds:
            subprocess.run(shlex.split(cmd), cwd=tmp_project, check=True)

        # Create virtual env
        venv_builder = venv.EnvBuilder(clear=True, with_pip=True)
        context = venv_builder.ensure_directories(tmp_project / '.venv')
        if hasattr(context, 'lib_path'):
            lib_path = context.lib_path
        else:
            lib_path = sysconfig.get_path('purelib', vars={
                'base': context.env_dir,
                'platbase': context.env_dir,
                'installed_base': context.env_dir,
                'installed_platbase': context.env_dir,
            })

        venv_builder.create(context.env_dir)

        # Install current_version of harissa
        subprocess.run(
            shlex.split(
                f'{context.env_exec_cmd} -m pip install {tmp_project}'
            ), 
            check=True
        )

        sys.path.insert(0, lib_path)

        output_root = Path(app.outdir).parent

        current_branch_semver = to_sem_ver(current_branch_version[1:])

        if current_version == current_branch:
            app.outdir = output_root / 'latest'
            current_semver = current_branch_semver
        else:
            current_semver = to_sem_ver(current_version[1:])

        config.version = current_semver

        config.html_title = project
        config.html_theme_options = {
            **config.html_theme_options,
            'switcher': {
                'json_url' : switcher_filename,
                'version_match': current_semver
            },
            'show_version_warning_banner': True,
            'navbar_start' : ['navbar-logo', 'version-switcher'],
            # 'navbar_align': 'left',
            # 'navbar_center': ['version-switcher', 'navbar-nav'],
            # 'navbar_end' : ['theme-switcher','navbar-icon-links', 'version-switcher']
        }
        
        stable_version = []
        new_switcher_data = []
        for data in switcher_data:
            data_version = data['version']
            new_data = {
                'name': data.get('name', data_version),
                'version': to_sem_ver(data_version[1:])
            }
            if data.get('preferred', False):
                new_data['url'] = f'/{module_name}/stable/'
                new_data['preferred'] = True
                stable_version.append(data_version)
            else:
                new_data['url'] = f'/{module_name}/{data_version}/'
            new_switcher_data.append(new_data)

        if len(stable_version) == 0 or len(stable_version) > 1:
            raise RuntimeError('Switcher file must have only 1 preferred version.')
        stable_version = stable_version[0]

        if current_version == stable_version:
            app.outdir = output_root / 'stable'

        config.version = Path(app.outdir).name
        config.release = current_semver

        config.sitemap_url_scheme = "{version}{link}"

        if current_branch:
            current_branch_data = {
                'name': current_branch_version, 
                'version': current_branch_semver,
                'url': f'/{module_name}/latest/'
            }

            new_switcher_data = [current_branch_data] + new_switcher_data
        
        with open(Path(app.srcdir) / switcher_filename, 'w') as fp:
            json.dump(new_switcher_data, fp, indent=4)

        if current_version == stable_version:
            logger.info('\033[1mBuilding Switcher.json\033[0m')
            with open(output_root / switcher_filename, 'w') as fp:
                json.dump(new_switcher_data, fp, indent=4)
            logger.info('Switcher generated.')

            logger.info('\033[1mBuilding root index.html\033[0m')
            with open(output_root / 'index.html', 'w') as fp:
                fp.write(root_index_content(html_baseurl))
            logger.info(f'Root index generated.\n' 
            f'It redirects to version {stable_version}')

            logger.info('\033[1mBuilding root sitemap\033[0m')
            with open(output_root / 'sitemapindex.xml', 'w') as fp:
                fp.write(root_sitemap_index_content(
                    html_baseurl,
                    [Path(data['url']).name for data in new_switcher_data]
                ))
            logger.info('Sitemap index generated.')
    
def clean_up(app, exception):
    if exception is not None:
        return 
    logger.info(f'\033[1m{sys.modules["harissa"]=}\033[0m')
    
    output = Path(app.outdir)
    current_version = app.config.smv_current_version
    
    if current_version:
        stable_version = list(filter(
            lambda data: data.get('preferred', False), 
            switcher_data
        ))[0]['version']

        if current_version == stable_version or current_version == current_branch:
            logger.info(f'\033[1mCleaning {current_version}\033[0m')
            rmtree(output.parent / current_version)

        logger.info('\033[1mCleaning extras\033[0m')
        for subpath in ['.doctrees', 'doctrees', 'objects.inv', '.buildinfo']:
            path = output / subpath
            if path.is_dir():
                rmtree(path)
            elif path.is_file():
                path.unlink()

    myst_nb_jupyter_execute_dir = output.parent / 'jupyter_execute'
    if myst_nb_jupyter_execute_dir.is_dir():
        logger.info('\033[1mCleaning jupyter_execute\033[0m')
        rmtree(myst_nb_jupyter_execute_dir)
    

def update_json_url(app, pagename, templatename, context, doctree):
    if app.config.smv_current_version:
        context['theme_switcher']['json_url'] = (
            f'/{module_name}/{switcher_filename}'
        )

def setup(app):
    if not (Path(app.srcdir) / 'notebooks' / 'index.md').exists():
        app.setup_extension('nbsphinx')
    else:
        conf_dir = app.confdir

        register_driver('copy_folder_only', CopyFolderOnly)
        app.connect('config-inited', setup_collections)
        app.setup_extension('sphinxcontrib.collections')
        app.connect('config-inited', reset_conf_dir(conf_dir))
        app.setup_extension('myst_nb')
        app.setup_extension('sphinxcontrib.mermaid')

    app.connect('config-inited', setup_multi_version)
    app.connect('html-page-context', update_json_url)
    app.connect('build-finished', clean_up)